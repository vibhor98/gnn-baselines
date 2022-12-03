"""Training pipeline for GraphML models."""

import argparse, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.data import register_data_args, load_data
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve, auc
from graphml_models import *
from conf import *
import networkx as nx


def get_model_and_config(name):
    name = name.lower()
    if name == 'gcn':
        return GCN, GCN_CONFIG
    elif name == 'gat':
        return GAT, GAT_CONFIG
    elif name == 'graphsage':
        return GraphSAGE, GRAPHSAGE_CONFIG
    elif name == 'appnp':
        return APPNP, APPNP_CONFIG
    elif name == 'tagcn':
        return TAGCN, TAGCN_CONFIG
    elif name == 'agnn':
        return AGNN, AGNN_CONFIG
    elif name == 'sgc':
        return SGC, SGC_CONFIG
    elif name == 'gin':
        return GIN, GIN_CONFIG
    elif name == 'chebnet':
        return ChebNet, CHEBNET_CONFIG

def evaluate(model, features, labels, mask):
    #model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def main(args):
    # load and preprocess dataset
    data, labels = dgl.load_graphs('kialo_dgl_graphs_sbert_joint.dgl')
    print('No. of Graphs:', len(data))
    num_examples = len(data)
    num_train = int(num_examples * 0.8)
    print('No. of Graphs in train set:', num_train)
    print('No. of Graphs in test set:', num_examples - num_train)

    train_sampler = SubsetRandomSampler(torch.arange(num_train))
    test_sampler = SubsetRandomSampler(torch.arange(num_train, num_examples))

    train_dataloader = GraphDataLoader(
        data, sampler=train_sampler, batch_size=5, drop_last=False)
    test_dataloader = GraphDataLoader(
        data, sampler=test_sampler, batch_size=5, drop_last=False)

    # print(data[0].ndata['label'].shape)

    #g = data[0]
    if args.gpu < 0:
        cuda = False
    else:
        cuda = True
        #g = g.to(args.gpu)
    # features = g.ndata['feat']
    # labels = g.ndata['label']
    # train_mask = g.ndata['train_mask']
    # val_mask = g.ndata['val_mask']
    # test_mask = g.ndata['test_mask']
    # in_feats = features.shape[1]
    in_feats = 384
    n_classes = 2
    # n_edges = data.graph.number_of_edges()
    # print("""----Data statistics------'
    #   #Edges %d
    #   #Classes %d
    #   #Train samples %d
    #   #Val samples %d
    #   #Test samples %d""" %
    #       (n_edges, n_classes,
    #           train_mask.int().sum().item(),
    #           val_mask.int().sum().item(),
    #           test_mask.int().sum().item()))

    # graph preprocess and calculate normalization factor
    # add self loop
    # if args.self_loop:
    #     g = g.remove_self_loop().add_self_loop()
    # n_edges = g.number_of_edges()

    # normalization
    # degs = g.in_degrees().float()
    # norm = torch.pow(degs, -0.5)
    # norm[torch.isinf(norm)] = 0
    # g.ndata['norm'] = norm.unsqueeze(1)

    # create GCN model
    GNN, config = get_model_and_config(args.model)
    model = GNN(in_feats,
                n_classes,
                *config['extra_args'])

    if cuda:
        model = model.cuda()

    print(model)

    loss_fcn = torch.nn.CrossEntropyLoss()

    # use optimizer
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=config['lr'],
                                 weight_decay=config['weight_decay'])

    # initialize graph
    for epoch in range(5):
        # if epoch >= 3:
        t0 = time.time()
        pred_all = []
        labels_all = []
        loss_all = []
        for batched_graph in train_dataloader:
        # for batched_graph in data[:num_train]:
            batched_graph = dgl.add_self_loop(batched_graph)
            batched_graph = dgl.add_reverse_edges(batched_graph)

            # forward
            logits = model(batched_graph, batched_graph.ndata['feat'].float())
            pred = logits.argmax(1)
            # print('labels:', set(batched_graph.ndata['label']))
            loss = loss_fcn(logits, batched_graph.ndata['label'])

            pred_all.extend(pred)
            labels_all.extend(batched_graph.ndata['label'])
            loss_all.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # if epoch >= 3:
        dur = time.time() - t0

        # acc = evaluate(model, features, labels, val_mask)
        acc = accuracy_score(pred_all, labels_all)
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f}"
             .format(epoch, dur, np.mean(loss_all), acc))

    # acc = evaluate(model, features, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(acc))
    num_correct = 0
    num_tests = 0
    pred_all = []
    labels_all = []
    pos_probs = []
    for batched_graph in test_dataloader:
    # for batched_graph in data[num_train:]:
        batched_graph = dgl.add_self_loop(batched_graph)
        logits = model(batched_graph, batched_graph.ndata['feat'].float())
        pred = logits.argmax(1)
        pos_probs.extend(logits[:, 1].detach().numpy())
        labels = batched_graph.ndata['label']
        num_correct += (pred == labels).sum().item()
        num_tests += len(labels)
        pred_all.extend(pred)
        labels_all.extend(labels)

    print("Test Accuracy {:.4f}".format(num_correct / num_tests))
    print("Precision:", precision_score(pred_all, labels_all))
    print("Recall:", recall_score(pred_all, labels_all))
    print("F1-score:", f1_score(pred_all, labels_all))
    print("Accuracy:", accuracy_score(pred_all, labels_all))
    P, R, _ = precision_recall_curve(labels_all, pos_probs)
    auc_score = auc(R, P)
    print("PR AUC:", auc_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Node classification for polarity prediction.')
    parser.add_argument("--model", type=str, default='gcn',
                        help='model to use, available models are gcn, gat, graphsage, gin,'
                             'appnp, tagcn, sgc, agnn')
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--self-loop", action='store_true',
            help="graph self-loop (default=False)")
    args = parser.parse_args()
    print(args)
    main(args)
