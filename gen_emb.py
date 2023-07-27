#pip install dgl
# conda install -c dglteam dgl-cuda10.2

import argparse
import os
import os.path as osp
import time

import dgl
import dgl.function as fn
import dgl.nn as dglnn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from torch.nn.parallel import DistributedDataParallel

from graph import *

root_base = os.getcwd()

class MultiLayerNeighborSampler(dgl.dataloading.BlockSampler): # overload
    def __init__(self, fanouts):
        super().__init__(len(fanouts))

        self.fanouts = fanouts

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout, prob='weight')
        return frontier

class MultiLayerDropoutSampler(dgl.dataloading.BlockSampler):
    def __init__(self, fanouts):
        super().__init__(len(fanouts))

        self.n_layers = len(fanouts)

    def sample_frontier(self, block_id, g, seed_nodes, *args, **kwargs):
        # Get all inbound edges to `seed_nodes`
        src, dst = dgl.in_subgraph(g, seed_nodes).all_edges()
        # Randomly select edges with a probability of p
        mask = dgl.in_subgraph(g, seed_nodes).edata['weight']
        src = src[mask]
        dst = dst[mask]
        # Return a new graph with the same nodes as the original graph as a
        # frontier
        frontier = dgl.graph((src, dst), num_nodes=g.number_of_nodes())
        return frontier

    def __len__(self):
        return self.n_layers

class SAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        super().__init__()
        self.init(in_feats, n_hidden, n_classes, n_layers, activation, dropout)

    def init(self, in_feats, n_hidden, n_classes, n_layers, activation, dropout):
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.layers = nn.ModuleList()
        if n_layers > 1:
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(1, n_layers - 1):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        else:
            self.layers.append(dglnn.SAGEConv(in_feats, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

    def forward(self, blocks, x):
        h = x
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h = layer(block, h)
            if l != len(self.layers) - 1:
                h = self.activation(h)
                h = self.dropout(h)
        return h

    def inference(self, g, x, device, batch_size, num_workers):
        """
        Inference with the GraphSAGE model on full neighbors (i.e. without neighbor sampling).
        g : the entire graph.
        x : the input of entire node set.
        The inference code is written in a fashion that it could handle any number of nodes and
        layers.
        """
        # During inference with sampling, multi-layer blocks are very inefficient because
        # lots of computations in the first few layers are repeated.
        # Therefore, we compute the representation of all nodes layer by layer.  The nodes
        # on each layer are of course splitted in batches.
        for l, layer in enumerate(self.layers):
            y = th.zeros(g.num_nodes(), self.n_hidden if l != len(self.layers) - 1 else self.n_classes)

            sampler = dgl.dataloading.MultiLayerFullNeighborSampler(1)
            dataloader = dgl.dataloading.NodeDataLoader(
                g,
                th.arange(g.num_nodes()).to(g.device),
                sampler,
                batch_size=batch_size,
                shuffle=True,
                drop_last=False,
                num_workers=num_workers)

            for input_nodes, output_nodes, blocks in tqdm.tqdm(dataloader):
                block = blocks[0]

                block = block.int().to(device)
                h = x[input_nodes].to(device)
                h = layer(block, h)
                if l != len(self.layers) - 1:
                    h = self.activation(h)
                    h = self.dropout(h)

                y[output_nodes] = h.cpu()

            x = y
        return y


class NegativeSampler(object):
    def __init__(self, g, k, neg_share=False):
        self.weights = g.in_degrees().float() ** 0.75
        self.k = k
        self.neg_share = neg_share

    def __call__(self, g, eids):
        src, _ = g.find_edges(eids)
        n = len(src)
        if self.neg_share and n % self.k == 0:
            dst = self.weights.multinomial(n, replacement=True)
            dst = dst.view(-1, 1, self.k).expand(-1, self.k, -1).flatten()
        else:
            dst = self.weights.multinomial(n*self.k, replacement=True)
        src = src.repeat_interleave(self.k)
        return src, dst

class CrossEntropyLoss(nn.Module):
    def forward(self, block_outputs, pos_graph, neg_graph):
        with pos_graph.local_scope():
            pos_graph.ndata['h'] = block_outputs
            pos_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            pos_score = pos_graph.edata['score']
        with neg_graph.local_scope():
            neg_graph.ndata['h'] = block_outputs
            neg_graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            neg_score = neg_graph.edata['score']

        score = th.cat([pos_score, neg_score])
        label = th.cat([th.ones_like(pos_score), th.zeros_like(neg_score)]).long()
        loss = F.binary_cross_entropy_with_logits(score, label.float())
        return loss

def evaluate(model, g, nfeat, device, node_ids, out_file):
    """
    Evaluate the model on the validation set specified by ``val_mask``.
    g : The entire graph.
    inputs : The features of all the nodes.
    val_mask : A 0-1 mask indicating which nodes do we actually compute the accuracy for.
    device : The GPU device to evaluate on.
    """
    model.eval()
    with th.no_grad():
        # single gpu
        if isinstance(model, SAGE):
            node_embeddings = model.inference(g, nfeat, device, args.batch_size, args.num_workers)
        # multi gpu
        else:
            node_embeddings = model.module.inference(g, nfeat, device, args.batch_size, args.num_workers)
    model.train()
    with open(out_file, "w") as f_out:
        f_out.write("{} {}\n".format(len(node_ids), args.emb_dim))
        for i in range(len(node_ids)):
            # if not (node_ids[i].startswith("i")):
            f_out.write("{} ".format(node_ids[i]))
            for j in range(args.emb_dim):
                f_out.write("{} ".format(node_embeddings[i][j]))
            f_out.write("\n")

#### Entry point
def run(proc_id, n_gpus, args, devices, data):
    # Unpack data
    device = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port='12345')
        world_size = n_gpus
        th.distributed.init_process_group(backend="nccl",
                                          init_method=dist_init_method,
                                          world_size=world_size,
                                          rank=proc_id)
    n_classes, g, num_nodes, node_ids = data
    nfeat = -2*th.rand((num_nodes, n_classes))+1
    #labels = g.ndata.pop('label')
    in_feats = nfeat.shape[1]

    # Create PyTorch DataLoader for constructing blocks
    n_edges = g.num_edges()
    train_seeds = th.arange(n_edges)

    # Create sampler
    sampler = MultiLayerNeighborSampler(
        [int(fanout) for fanout in args.fan_out.split(',')])
    dataloader = dgl.dataloading.EdgeDataLoader(
        g, train_seeds, sampler, exclude='reverse_id',
        # For each edge with ID e in Reddit dataset, the reverse edge is e ± |E|/2.
        reverse_eids=th.cat([
            th.arange(n_edges // 2, n_edges),
            th.arange(0, n_edges // 2)]).to(train_seeds),
        negative_sampler=NegativeSampler(g, args.num_negs, args.neg_share),
        device=device,
        #use_ddp=n_gpus > 1,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers)

    # Define model and optimizer
    model = SAGE(in_feats, args.num_hidden, args.emb_dim, args.num_layers, F.relu, args.dropout)
    model = model.to(device)
    if n_gpus > 1:
        model = DistributedDataParallel(model, device_ids=[device], output_device=device)
    loss_fcn = CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    out_file = osp.abspath(osp.join(root_base, args.output_path.format(args.site_id), "embedding/dim_{}.txt".format(args.emb_dim)))

    if not osp.exists(out_file):
        # Training loop
        avg = 0
        iter_pos = []
        iter_neg = []
        iter_d = []
        iter_t = []
        best_eval_acc = 0
        best_test_acc = 0
        for epoch in range(args.num_epochs):
            if n_gpus > 1:
                dataloader.set_epoch(epoch)
            tic = time.time()

            # Loop over the dataloader to sample the computation dependency graph as a list of
            # blocks.

            tic_step = time.time()
            for step, (input_nodes, pos_graph, neg_graph, blocks) in enumerate(dataloader):
                batch_inputs = nfeat[input_nodes].to(device)
                d_step = time.time()

                pos_graph = pos_graph.to(device)
                neg_graph = neg_graph.to(device)
                blocks = [block.int().to(device) for block in blocks]
                # Compute loss and prediction
                batch_pred = model(blocks, batch_inputs)
                loss = loss_fcn(batch_pred, pos_graph, neg_graph)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                t = time.time()
                pos_edges = pos_graph.num_edges()
                neg_edges = neg_graph.num_edges()
                iter_pos.append(pos_edges / (t - tic_step))
                iter_neg.append(neg_edges / (t - tic_step))
                iter_d.append(d_step - tic_step)
                iter_t.append(t - d_step)
                if step % args.log_every == 0:
                    # gpu_mem_alloc = th.cuda.max_memory_allocated() / 1000000 if th.cuda.is_available() else 0
                    print('[{}]Epoch {:05d} | Step {:05d} | Loss {:.4f} | Speed (samples/sec) {:.4f}|{:.4f} | Load {:.4f}| train {:.4f}'.format(
                        proc_id, epoch, step, loss.item(), np.mean(iter_pos[3:]), np.mean(iter_neg[3:]), np.mean(iter_d[3:]), np.mean(iter_t[3:])))
                tic_step = time.time()

            toc = time.time()
            if proc_id == 0:
                print('Epoch Time(s): {:.4f}'.format(toc - tic))
            if epoch >= 5:
                avg += toc - tic
            if n_gpus > 1:
                th.distributed.barrier()

        if proc_id == 0:
            evaluate(model, g, nfeat, device, node_ids, out_file)
            print('Avg epoch time: {}'.format(avg / (epoch - 4)))
    else:
        print("Embedding file exists already")

def main(args, devices):
    n_classes = args.emb_dim
    g = Graph()
    g.read_edgelist(filename=osp.abspath(osp.join(root_base, args.output_path.format(args.site_id), "raw_data/prune_{}.edgelist".format(args.site_id))), weighted=True)
    node_ids = list(g.G.nodes())

    g_init = dgl.from_networkx(g.G, edge_attrs=['weight'])

    # g_init = dgl.from_networkx(g.G)

    # Create csr/coo/csc formats before launching training processes with multi-gpu.
    # This avoids creating certain formats in each sub-process, which saves memory and CPU.
    #g.create_formats_()
    # Pack data
    data = n_classes, g_init, g_init.num_nodes(), node_ids

    n_gpus = len(devices)
    if devices[0] == -1:
        run(0, 0, args, ['cpu'], data)
    else:
        run(0, n_gpus, args, devices, data)

argparser = argparse.ArgumentParser()
argparser.add_argument("--gpu", type=str, default='1')
argparser.add_argument('--num-epochs', type=int, default=50)
argparser.add_argument('--num-hidden', type=int, default=32)
argparser.add_argument('--num-layers', type=int, default=2)
argparser.add_argument('--num-negs', type=int, default=4)
argparser.add_argument('--neg-share', default=False, action='store_true')
argparser.add_argument('--fan-out', type=str, default='10,25')
argparser.add_argument('--batch-size', type=int, default=200)
argparser.add_argument('--log-every', type=int, default=200)
argparser.add_argument('--save-every', type=int, default=100)
argparser.add_argument('--lr', type=float, default=0.003)
argparser.add_argument('--dropout', type=float, default=0.2)
argparser.add_argument('--num-workers', type=int, default=4)
# added
argparser.add_argument('--site_id', type=str, default="5cd56bb5e2acfd2d33b62b23")
argparser.add_argument('--emb_dim', type=int, default=32)
argparser.add_argument('--output_path', type=str, default="./data/output/{}")
args = argparser.parse_args()

devices = list(map(int, args.gpu.split(',')))

buildings = os.listdir('./data/input')

for building in buildings:
    print (f'building: {building}')
    args.site_id = building
    main(args, devices)

