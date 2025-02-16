import os
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import random

import torch
import torch.nn as nn

import transformers

from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from torch_geometric.data import Data
import dgl

from model.model_na import AGG
from model.model_bp import AGG_BP, degree
from model.model_bp_appr import AGG_BP_APPR

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def convert_dict_to_args(_dict):
    parser = argparse.ArgumentParser()
    for key, value in _dict.items():
        parser.add_argument(f"--{key}", default=value, type=type(value))
    return parser.parse_args([])

class Runner():
    def __init__(self, args):
        self.args = args
        self.num_class_dict = {'cora': 7, 'citeseer': 6, 'pubmed': 3,
                                'bookhis': 12, 'bookchild': 24, 'sportsfit': 13,  'wikics': 10, 
                                'cornell': 5, 'texas': 5, 'washington': 5, 'wisconsin': 5}
        
        # below is the prediction from GPT-4o-mini, which can be implemented via pred_h.py
        self.H_dict = {'cora': 0.70, 'citeseer': 0.81, 'pubmed': 0.81,
                       'bookhis': 0.73, 'bookchild': 0.35, 'sportsfit': 0.81,  'wikics': 0.52, 
                       'cornell': 0.05, 'texas': 0.04, 'wisconsin': 0.06, 'washington': 0.02}
        self.num_class = self.num_class_dict[self.args.dataset]
        
    def run(self):
        self.device = torch.device('cuda:'+str(self.args.device))
        graph_data, embs = self.prepare_data()
        raws = []
        nas =[]
        bps = []
        bp_apprs = []
        raws_f1 = []
        nas_f1 = []
        bps_f1 = []
        bp_apprs_f1 = []
        for seed in range(42, 52):
            if self.args.dataset not in ['cornell', 'texas', 'wisconsin', 'washington']:
                label_embs = self.get_label_embs_k_shot(graph_data, embs, self.args.k, seed)
            else:
                label_embs, graph_data = self.get_label_embs_k_shot_school(graph_data, embs, self.args.k, seed)
            self.prepare_AGG(0)
            accuracy_raw, f1_raw = self.test_generalization(graph_data, embs, label_embs)
            self.prepare_AGG(1)
            accuracy_na, f1_na = self.test_generalization(graph_data, embs, label_embs)
            self.prepare_AGG_bp(self.H_dict[self.args.dataset])
            accuracy_bp, f1_bp = self.test_generalization_bp(graph_data, embs, label_embs)
            if self.args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
                self.prepare_AGG(-0.5)
            else:
                self.prepare_AGG(0.5)
            accuracy_bp_appr, f1_bp_appr = self.test_generalization_bp_appr(graph_data, embs, label_embs)
            raws.append(accuracy_raw.item())
            raws_f1.append(f1_raw.item())
            nas.append(accuracy_na.item())
            nas_f1.append(f1_na.item())
            bps.append(accuracy_bp.item())
            bps_f1.append(f1_bp.item())
            bp_apprs.append(accuracy_bp_appr.item())
            bp_apprs_f1.append(f1_bp_appr.item())
        print(f'Vanilla encoder:   Accuracy: {round(np.mean(raws)*100,2)} +- {round(np.std(raws)*100,2)} , F1: {round(np.mean(raws_f1)*100,2)} +- {round(np.std(raws_f1)*100,2)}')
        print(f'Neighborhood Aggregation: Accuracy:  {round(np.mean(nas)*100,2)} +- {round(np.std(nas)*100,2)}, F1: {round(np.mean(nas_f1)*100,2)} +- {round(np.std(nas)*100,2)}')
        print(f'BP Algorithm: Accuracy: {round(np.mean(bps)*100,2)} +- {round(np.std(bps)*100,2)}, F1: {round(np.mean(bps_f1)*100,2)} +- {round(np.std(bps_f1)*100,2)}')
        print(f'BP (appr.) Algorithm: Accuracy: {round(np.mean(bp_apprs)*100,2)} +- {round(np.std(bp_apprs)*100,2)}, F1: {round(np.mean(bp_apprs_f1)*100,2)} +- {round(np.std(bp_apprs_f1)*100,2)}')

    def prepare_AGG_bp_appr(self, H):
        self.model = AGG(H)
        self.model = self.model.to(self.device)

    def prepare_AGG_bp(self, H):
        H_matrix = (1 - H) * torch.ones((self.num_class, self.num_class))  
        H_matrix.fill_diagonal_(H)
        H_matrix = H_matrix.to(self.device)
        self.model = AGG_BP(H_matrix)
        self.model = self.model.to(self.device)
    
    def prepare_AGG(self, weight):
        self.model = AGG(weight)
        self.model = self.model.to(self.device)

    def prepare_data(self):
        if self.args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
            dgl_graph = dgl.load_graphs(f'./dataset/{self.args.dataset}/{self.args.dataset.capitalize()}.pt')[0][0]
            edge_index = torch.stack(dgl_graph.edges())
            graph_data = Data(edge_index = edge_index, y = dgl_graph.ndata['label'])
            graph_data.test_id = torch.arange(len(graph_data.y))
            graph_data.train_id = torch.arange(len(graph_data.y))
        else:
            graph_data = torch.load(f'./dataset/{self.args.dataset}/processed_data.pt')
        embs = torch.load(f'./dataset/{self.args.dataset}/{self.args.encoder}_x.pt')
        if self.args.dataset in ['wikics']:
            pkg = torch.where(graph_data.test_mask)
            pkg2 = torch.where(graph_data.train_mask)
            graph_data.test_id = pkg[0]
            graph_data.train_id = pkg2[0]
        return graph_data, embs

    def test_generalization(self, graph_data, embs, label_embs):
        labels = graph_data.y.to(self.device)
        if torch.is_tensor(label_embs):
            label_embs = label_embs.to(self.device)
        else:
            label_embs = torch.tensor(label_embs).to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        feature = embs.to(self.device)
        agg_feature = self.model(feature, edge_index)
        agg_feature_normed = F.normalize(agg_feature, p=2, dim = 1)
        acc_agg_feature, f1 = self.match_feature(agg_feature_normed, label_embs, labels, graph_data.test_id)
        return acc_agg_feature, f1

    def match_feature(self, feature, label_feature, labels, test_id):
        similarity_matrix = torch.mm(feature, label_feature.T)
        max_indices = torch.argmax(similarity_matrix, dim=1)
        same_flag = (labels==max_indices)
        acc = torch.sum(same_flag[test_id])/test_id.shape[0]
        f1_macro = f1_score(labels.cpu()[test_id], max_indices.cpu()[test_id], average='macro') 
        return acc, f1_macro

    def test_generalization_bp(self, graph_data, embs, label_embs):
        H = self.H_dict[self.args.dataset]
        N = graph_data.y.shape[0]
        labels = graph_data.y.to(self.device)
        if torch.is_tensor(label_embs):
            label_embs = label_embs.to(self.device)
        else:
            label_embs = torch.tensor(label_embs).to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        edge_index, od = self.sort_edge(N, edge_index, True)
        _, edge_rv = self.sort_edge(N, edge_index, False)
        
        feature = embs.to(self.device)
        similarity_matrix = torch.mm(feature, label_embs.T)
        probability_matrix = F.softmax(similarity_matrix/0.025, dim=1)
        log_probability_matrix = torch.log(probability_matrix)
        deg = degree(edge_index[1], N)
        edge_weight = torch.ones(edge_index.shape[1], dtype=torch.float32)
        deg = deg.to(self.device)
        edge_weight = edge_weight.to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        edge_rv = edge_rv.to(self.device)
        mix_likelihood = self.model(log_probability_matrix, edge_index, edge_weight, edge_rv, deg)
        max_indices = torch.argmax(mix_likelihood, dim=1)
        same_flag = (labels==max_indices)
        test_id = graph_data.test_id
        acc = torch.sum(same_flag[test_id])/test_id.shape[0]
        f1_macro = f1_score(labels.cpu()[test_id], max_indices.cpu()[test_id], average='macro') 
        return acc, f1_macro
        
    def test_generalization_bp_appr(self, graph_data, embs, label_embs):
        labels = graph_data.y.to(self.device)
        if torch.is_tensor(label_embs):
            label_embs = label_embs.to(self.device)
        else:
            label_embs = torch.tensor(label_embs).to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        feature = embs.to(self.device)
        similarity_matrix = torch.mm(feature, label_embs.T)
        probability_matrix = F.softmax(similarity_matrix/0.01, dim=1)
        if self.args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
            weight = -0.5
        else:
            weight = 0.5
        mix_likelihood = self.model(probability_matrix, edge_index)
        max_indices = torch.argmax(mix_likelihood, dim=1)
        same_flag = (labels==max_indices)
        test_id = graph_data.test_id
        acc = torch.sum(same_flag[test_id])/test_id.shape[0]
        f1_macro = f1_score(labels.cpu()[test_id], max_indices.cpu()[test_id], average='macro') 
        return acc, f1_macro
        
    def get_label_embs_k_shot(self, graph_data, embs, mode, seed):
        # first get all the training data according to each class
        embs = embs[graph_data.train_id]
        labels = graph_data.y[graph_data.train_id]
        label_embs = torch.zeros((self.num_class, embs.shape[1]))
        if mode == 'full':
            for class_idx in range(self.num_class):
                mask = (labels == class_idx) 
                mean_feature = embs[mask].mean(dim=0).reshape(1, -1)
                mean_feature = F.normalize(mean_feature, p=2, dim = 1)
                label_embs[class_idx] = mean_feature.reshape(-1)
            return label_embs
        else:
            top_k = mode
            for class_idx in range(self.num_class):
                mask = (labels == class_idx) 
                true_indices = torch.where(mask)[0]
                torch.manual_seed(seed)
                sampled_indices = true_indices[torch.randint(0, len(true_indices), (top_k,))]
                print(f'class {class_idx} sampled indices: {sampled_indices}')
                mean_feature = embs[sampled_indices].mean(dim=0).reshape(1, -1)
                mean_feature = F.normalize(mean_feature, p=2, dim = 1)
                label_embs[class_idx] = mean_feature.reshape(-1)
            return label_embs
        
    def get_label_embs_k_shot_school(self, graph_data, embs, mode, seed):
        # first get all the training data according to each class
        embs = embs[graph_data.train_id]
        labels = graph_data.y[graph_data.train_id]
        label_embs = torch.zeros((self.num_class, embs.shape[1]))
        graph_data.test_id = []
        top_k = mode
        for class_idx in range(self.num_class):
            mask = (labels == class_idx) 
            true_indices = torch.where(mask)[0]
            torch.manual_seed(seed)
            sampled_indices = true_indices[torch.randint(0, len(true_indices), (top_k,))]
            print(f'class {class_idx} sampled indices: {sampled_indices}')
            tests = ~torch.isin(true_indices, sampled_indices)
            rest = true_indices[tests]
            graph_data.test_id.extend(rest)
            mean_feature = embs[sampled_indices].mean(dim=0).reshape(1, -1)
            mean_feature = F.normalize(mean_feature, p=2, dim = 1)
            label_embs[class_idx] = mean_feature.reshape(-1)
        graph_data.test_id = torch.tensor(graph_data.test_id)
        return label_embs, graph_data

    def sort_edge(self, num_nodes, edge_index, sort_by_row=True):
        assert (edge_index.shape[1] == 0) or (0 <= edge_index.min()) and (edge_index.max() <= num_nodes-1)
        idx = edge_index[1-int(sort_by_row)]*num_nodes+edge_index[int(sort_by_row)]
        perm = idx.argsort()
        return edge_index[:, perm], perm


def main():
    parser = argparse.ArgumentParser(description='few shot inference')
    # hardware and general
    parser.add_argument('--device', dest = 'device', type = str, default = '0', help = 'when use ray can set multiple devices')
    # data
    parser.add_argument('--task', dest = 'task', type = str, default = 'nc', help = 'nc, lp')
    parser.add_argument('--dataset', dest = 'dataset', type = str, default = 'cora', help = 'cora')
    parser.add_argument('--encoder', dest = 'encoder', type = str, default = 'roberta', help = 'roberta, sbert')
    parser.add_argument('--k', dest = 'k', type = int, default = 10)
    args = parser.parse_args()

    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()

    