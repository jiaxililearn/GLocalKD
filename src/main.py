# -*- coding: utf-8 -*-
"""
@author: MaRongrong
@editor: Jiaxi Li
"""
import os
import numpy as np
from sklearn.utils.random import sample_without_replacement
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.svm import OneClassSVM
import argparse
import load_data
import networkx as nx
from GCN_embedding import GcnEncoderGraph_teacher, GcnEncoderGraph_student
import torch
import torch.nn as nn
import time
import GCN_embedding
from torch.autograd import Variable
from graph_sampler import GraphSampler
from numpy.random import seed
import random
import matplotlib.pyplot as plt
import copy
import torch.nn.functional as F
from sklearn.manifold import TSNE
from matplotlib import cm
from tdc.utils import retrieve_label_name_list
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm


def arg_parse():
    parser = argparse.ArgumentParser(description="GLocalKD Arguments.")
    parser.add_argument(
        "--datadir",
        dest="datadir",
        default="dataset",
        help="Directory where benchmark is located",
    )
    parser.add_argument("--DS", dest="DS", default="AIDS", help="dataset name")
    parser.add_argument(
        "--max-nodes",
        dest="max_nodes",
        type=int,
        default=0,
        help="Maximum number of nodes (ignore graghs with nodes exceeding the number.",
    )
    parser.add_argument(
        "--clip", dest="clip", default=0.1, type=float, help="Gradient clipping."
    )
    parser.add_argument(
        "--num_epochs",
        dest="num_epochs",
        default=150,
        type=int,
        help="total epoch number",
    )
    parser.add_argument(
        "--batch-size", dest="batch_size", default=300, type=int, help="Batch size."
    )
    parser.add_argument(
        "--sample-size", dest="sample_size", default=300, type=int, help="Sample size."
    )
    parser.add_argument(
        "--hidden-dim",
        dest="hidden_dim",
        default=512,
        type=int,
        help="Hidden dimension",
    )
    parser.add_argument(
        "--output-dim",
        dest="output_dim",
        default=256,
        type=int,
        help="Output dimension",
    )
    parser.add_argument(
        "--num-gc-layers",
        dest="num_gc_layers",
        default=3,
        type=int,
        help="Number of graph convolution layers before each pooling",
    )
    parser.add_argument(
        "--num-node-types",
        dest="num_node_types",
        default=1,
        type=int,
        help="Number of node types",
    )
    parser.add_argument(
        "--nobn",
        dest="bn",
        action="store_const",
        const=False,
        default=True,
        help="Whether batch normalization is used",
    )
    parser.add_argument(
        "--dropout", dest="dropout", default=0.3, type=float, help="Dropout rate."
    )
    parser.add_argument(
        "--lr", dest="lr", default=0.0001, type=float, help="Learning Rate"
    )
    parser.add_argument(
        "--nobias",
        dest="bias",
        action="store_const",
        const=False,
        default=True,
        help="Whether to add bias. Default to True.",
    )
    parser.add_argument(
        "--feature", dest="feature", default="default", help="use what node feature"
    )
    parser.add_argument("--seed", dest="seed", type=int, default=1, help="seed")
    parser.add_argument(
        "--cpu", dest="cpu", type=bool, default=False, help="using CPU or not"
    )
    parser.add_argument(
        "--fix-train-test", dest="fix_train_test", type=bool, default=False, help="use a fixed train/test set"
    )
    parser.add_argument(
        "--sagemaker", dest="sagemaker", type=bool, default=False, help="if using sagemaker training"
    )
    return parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(dataset, data_test_loader, model_teacher, model_student, args):
    device = 'cpu' if args.cpu else 'cuda'

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model_student.parameters()), lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
    epochs = []
    auroc_final = 0
    for epoch in range(args.num_epochs):
        total_time = 0
        total_loss = 0.0
        model_student.train()

        for batch_idx, data in tqdm(enumerate(dataset)):
            begin_time = time.time()
            model_student.zero_grad()
            adj = Variable(data["adj"].float(), requires_grad=False).to(device)
            h0 = Variable(data["feats"].float(), requires_grad=False).to(device)
            node_types = Variable(data["node_types"].int(), requires_grad=False)#.to(device)

            embed_node, embed = model_student(h0, adj, node_types)
            # print('---> finished student <---')
            embed_teacher_node, embed_teacher = model_teacher(h0, adj, node_types)
            # print('---> finished teacher <---')
            embed_teacher = embed_teacher.detach()
            embed_teacher_node = embed_teacher_node.detach()
            loss_node = (
                torch.mean(
                    F.mse_loss(embed_node, embed_teacher_node, reduction="none"), dim=2
                )
                .mean(dim=1)
                .mean(dim=0)
            )
            loss = (
                F.mse_loss(embed, embed_teacher, reduction="none")
                .mean(dim=1)
                .mean(dim=0)
            )
            forward_time = time.time() - begin_time

            loss = loss + loss_node

            loss.backward(loss.clone().detach())
            nn.utils.clip_grad_norm_(model_student.parameters(), args.clip)
            optimizer.step()
            scheduler.step()
            total_loss += loss
            elapsed = time.time() - begin_time
            total_time += elapsed
            print(f"Batch Loss: {loss}; Batch Time: {elapsed}; Forward Time: {forward_time}")

            # clear gpu cache
            torch.cuda.empty_cache()

        if (epoch + 1) % 2 == 0 and epoch > 0:
            epochs.append(epoch)
            model_student.eval()
            loss = []
            y = []
            emb = []

            for batch_idx, data in enumerate(data_test_loader):
                adj = Variable(data["adj"].float(), requires_grad=False).to(device)
                h0 = Variable(data["feats"].float(), requires_grad=False).to(device)
                node_types = Variable(data["node_types"].int(), requires_grad=False).to(device)

                embed_node, embed = model_student(h0, adj, node_types)
                embed_teacher_node, embed_teacher = model_teacher(h0, adj, node_types)
                loss_node = torch.mean(
                    F.mse_loss(embed_node, embed_teacher_node, reduction="none"), dim=2
                ).mean(dim=1)
                loss_graph = F.mse_loss(embed, embed_teacher, reduction="none").mean(
                    dim=1
                )
                loss_ = loss_graph + loss_node
                loss_ = np.array(loss_.cpu().detach())
                loss.append(loss_)
                if data["label"] == 0:
                    y.append(1)
                else:
                    y.append(0)
                emb.append(embed.cpu().detach().numpy())

            label_test = []
            for loss_ in loss:
                label_test.append(loss_)
            label_test = np.array(label_test)

            fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
            test_roc_ab = auc(fpr_ab, tpr_ab)

            precision, recall, pr_thresholds = precision_recall_curve(
                y, label_test
            )
            ap = auc(recall, precision)
            print(
                "semi-supervised abnormal detection: auroc_ab: {}".format(test_roc_ab)
            )
            print(f"\tAUC:{test_roc_ab}; Avg Precision:{ap};")

        if epoch == (args.num_epochs - 1):
            auroc_final = test_roc_ab
        print(f"Epoch Loss: {total_loss}; Epoch Time: {total_time};")

    return auroc_final

def read_train_test_index(datadir, dataname):
    prefix = os.path.join(datadir, dataname)
    with open(f'{prefix}/model_gid_list_train.txt') as fin:
        train_gids = [int(i) for i in fin.read().strip().split()]
    with open(f'{prefix}/model_gid_list_eval.txt') as fin:
        eval_gids = [int(i) for i in fin.read().strip().split()]

    return zip([train_gids], [eval_gids])

if __name__ == "__main__":
    args = arg_parse()
    DS = args.DS
    setup_seed(args.seed)

    device = 'cpu' if args.cpu else 'cuda'

    # Resolve Args
    if args.sagemaker:
        args.datadir = os.environ['SM_CHANNEL_TRAIN']

    graphs = load_data.read_graphfile(args.datadir, args.DS, max_nodes=args.max_nodes)
    datanum = len(graphs)
    if args.max_nodes == 0:
        max_nodes_num = max([G.number_of_nodes() for G in graphs])
    else:
        max_nodes_num = args.max_nodes
    print(datanum)
    graphs_label = [graph.graph["label"] for graph in graphs]

    # if fix the train / test sets
    if args.fix_train_test:
        train_test_index = read_train_test_index(args.datadir, args.DS)

    else:
        kfd = StratifiedKFold(n_splits=5, random_state=args.seed, shuffle=True)
        train_test_index = kfd.split(graphs, graphs_label)

    result_auc = []
    for k, (train_index, test_index) in enumerate(train_test_index):

        # Update: opt mem to store same set of graphs
        # graphs_train_ = [graphs[i] for i in train_index]
        # graphs_test = [graphs[i] for i in test_index]

        # graphs_train = []
        # for graph in graphs_train_:
        #     if graph.graph["label"] != 0:
        #         graphs_train.append(graph)

        num_train = len(train_index)
        num_test = len(test_index)
        print(num_train, num_test)

        dataset_sampler_train = GraphSampler(
            # graphs_train,
            graphs,
            train_index,
            G_list=None,
            features=args.feature,
            normalize=False,
            max_num_nodes=max_nodes_num,
            sample_size=args.sample_size
        )
        # print(f'dataset_sampler_train sample 1: {dataset_sampler_train[0]}')

        model_teacher = GCN_embedding.GcnEncoderGraph_teacher(
            dataset_sampler_train.feat_dim,
            args.hidden_dim,
            args.output_dim,
            2,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        ).to(device)
        for param in model_teacher.parameters():
            param.requires_grad = False

        model_student = GCN_embedding.GcnEncoderGraph_student(
            dataset_sampler_train.feat_dim,
            args.hidden_dim,
            args.output_dim,
            2,
            args.num_gc_layers,
            bn=args.bn,
            args=args,
        ).to(device)

        data_train_loader = torch.utils.data.DataLoader(
            dataset_sampler_train, shuffle=True, batch_size=args.batch_size
        )

        dataset_sampler_test = GraphSampler(
            # graphs_test,
            graphs,
            test_index,
            G_list=None,
            features=args.feature,
            normalize=False,
            max_num_nodes=max_nodes_num,
        )
        data_test_loader = torch.utils.data.DataLoader(
            dataset_sampler_test, shuffle=False, batch_size=1
        )
        result = train(
            data_train_loader, data_test_loader, model_teacher, model_student, args
        )
        result_auc.append(result)

    result_auc = np.array(result_auc)
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    print("auroc{}, average: {}, std: {}".format(result_auc, auc_avg, auc_std))
