import os
import copy
import pathlib
import argparse
from tqdm import tqdm
import re
import random
from collections import defaultdict

import pandas as pd
import scipy.sparse as sp
import numpy as np
from sklearn.metrics import f1_score
import pickle as pkl

import torch
import torch.nn.functional as F
import torch.nn as nn

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import login
import openai
from openai import OpenAI

from torch_geometric.nn import MessagePassing
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


def index_dict(lst, k):
    index_map = defaultdict(list)
    for i, num in enumerate(lst):
        if 0 <= num <= k:
            index_map[num].append(i)
    return dict(index_map)

def find_central_embeddings(embeddings, top_n=10):
    embeddings = np.array(embeddings)
    centroid = np.mean(embeddings, axis=0)  
    distances = np.linalg.norm(embeddings - centroid, axis=1)  
    central_indices = np.argsort(distances)[:top_n]  
    central_embeddings = embeddings[central_indices]  
    central_avg = np.mean(central_embeddings, axis=0)
    norm = np.linalg.norm(central_avg)
    return central_avg/norm, central_indices
        
class Runner():
    def __init__(self, args):
        self.args = args
        self.num_class_dict = {'cora': 7, 'citeseer': 6, 'pubmed': 3, 
                                'bookhis': 12, 'bookchild': 24, 'sportsfit': 13, 'wikics': 10, 
                                'cornell': 5, 'texas': 5, 'washington': 5, 'wisconsin': 5}

        # below is the prediction from GPT-4o-mini, which can be implemented via pred_h.py
        self.H_dict = {'cora': 0.70, 'citeseer': 0.81, 'pubmed': 0.81,
                       'bookhis': 0.73, 'bookchild': 0.35, 'sportsfit': 0.81,  'wikics': 0.52, 
                       'cornell': 0.05, 'texas': 0.04, 'wisconsin': 0.06, 'washington': 0.02,}
        self.num_class = self.num_class_dict[self.args.dataset]
            
    def run(self):
        self.train_folder = './results/'+ str(self.args.task)+'_'+str(self.args.dataset)+'/'+str(self.args.model)+'/llm_baseline/'
        methods =['Raw', 'w/ NA', 'w/ BP', 'w/ BP (appr.)']
        metrics = ["accuracy", "F1"]

        # load the test data and the text of labels
        graph_data, embs = self.prepare_data()
        test_label = graph_data.y[graph_data.test_id]
        test_embs = embs[graph_data.test_id]
        labels = self.load_labels(self.args.dataset)

        # read the prediction from LLMs and randomly sample 20k, repeat 30 times with different seeds
        pth = self.train_folder+f'response_{0}.pkl'
        with open(pth, "rb") as file:
            loaded_list = pkl.load(file)
        pred_list = self.catch_answer(loaded_list, labels)
        pred_list = np.array(pred_list)
        
        raw_acc = []
        raw_F1 = []
        na_acc = []
        na_F1 = []
        bp_acc = []
        bp_F1 = []
        bp_appr_acc = []
        bp_appr_F1 = []
        for seed in tqdm(range(42,72)):
            random.seed(seed)
            k = 20
            sampled_elements = random.sample(range(len(pred_list)), self.num_class*k)
            tmp_pred_list = [pred_list[i] for i in sampled_elements]
            tmp_embs = test_embs[sampled_elements]
            result = index_dict(tmp_pred_list, self.num_class)
            
            ee = []
            for i in range(self.num_class):
                if i in result.keys():
                    ee.append(tmp_embs[result[i]])
            
            # find the central 10 embeddings, use their average embedding as class embedding
            new_label_embs = []
            for idx, ee_ in enumerate(ee):
                new_label_ee, cent_idx = find_central_embeddings(ee_)
                new_label_embs.append(new_label_ee.reshape(-1))
            new_label_embs = np.array(new_label_embs)

            # perform algorithms for inference
            self.device = torch.device(f'cuda:{self.args.device}')
            for weight in [0,1]:
                # weight=0: vanilla encoder
                # weight=1: neighborhood aggregation (NA)
                self.prepare_AGG(weight)
                accuracy, f1 = self.test_generalization(graph_data, embs, new_label_embs, list(result.keys()))
                if weight == 0:
                    raw_acc.append(accuracy.item())
                    raw_F1.append(f1.item())
                elif weight ==1:
                    na_acc.append(accuracy.item())
                    na_F1.append(f1.item())
            
            # test BP algorithm
            accuracy_bp, f1_bp = self.test_generalization_bp(graph_data, embs, new_label_embs, list(result.keys()))
            bp_acc.append(accuracy_bp.item())
            bp_F1.append(f1_bp.item())
            
            # test BP (appr.) algorithm
            accuracy_bp, f1_bp = self.test_generalization_bp_appr(graph_data, embs, new_label_embs, list(result.keys()))
            bp_appr_acc.append(accuracy_bp.item())
            bp_appr_F1.append(f1_bp.item())

        print(f'Vanilla encoder:   Accuracy: {round(np.mean(raw_acc)*100,2)} +- {round(np.std(raw_acc)*100,2)}, F1: {round(np.mean(raw_F1)*100,2)} +- {round(np.std(raw_F1)*100,2)}')
        print(f'Neighborhood Aggregation: Accuracy:  {round(np.mean(na_acc)*100,2)} +- {round(np.std(na_acc)*100,2)}, F1: {round(np.mean(na_F1)*100,2)} +- {round(np.std(na_F1)*100,2)}')
        print(f'BP Algorithm: Accuracy: {round(np.mean(bp_acc)*100,2)} +- {round(np.std(bp_acc)*100,2)}, F1: {round(np.mean(bp_F1)*100,2)} +- {round(np.std(bp_F1)*100,2)}')
        print(f'BP (appr.) Algorithm: Accuracy: {round(np.mean(bp_appr_acc)*100,2)} +- {round(np.std(bp_appr_acc)*100,2)}, F1: {round(np.mean(bp_appr_F1)*100,2)} +- {round(np.std(bp_appr_F1)*100,2)}')

    def prepare_AGG(self, weight):
        self.model = AGG(weight)
        self.model = self.model.to(self.device)

    def prepare_AGG_bp(self, H, dims):
        H_matrix = (1 - H) * torch.ones((dims, dims))  
        H_matrix.fill_diagonal_(H)
        H_matrix = H_matrix.to(self.device)
        self.model = AGG_BP(H_matrix)
        self.model = self.model.to(self.device)

    def prepare_AGG_bp_appr(self, weight):
        self.model = AGG_BP_APPR(weight)
        self.model = self.model.to(self.device)

    def test_generalization(self, graph_data, embs, label_embs, key_list):
        labels = graph_data.y.to(self.device)
        if torch.is_tensor(label_embs):
            label_embs = label_embs.to(self.device)
        else:
            label_embs = torch.tensor(label_embs).to(self.device)
        edge_index = graph_data.edge_index.to(self.device)
        feature = embs.to(self.device)
        agg_feature = self.model(feature, edge_index)
        agg_feature_normed = F.normalize(agg_feature, p=2, dim = 1)
        acc_agg_feature, f1_macro = self.match_feature(agg_feature_normed, label_embs, labels, graph_data.test_id, key_list)
        return acc_agg_feature, f1_macro

    def match_feature(self, feature, label_feature, labels, test_id, key_list):
        similarity_matrix = torch.mm(feature, label_feature.T)
        max_indices = torch.argmax(similarity_matrix, dim=1)
        mapping = {}
        key_list = sorted(key_list)
        for i in range(label_feature.shape[0]):
            mapping[i] = key_list[i]
        new_max_indices = list(map(lambda x: mapping[x], max_indices.cpu().numpy()))
        new_max_indices = torch.tensor(new_max_indices).to(self.device)
        same_flag = (labels==new_max_indices)
        acc = torch.sum(same_flag[test_id])/test_id.shape[0]
        f1_macro = f1_score(labels.cpu()[test_id], max_indices.cpu()[test_id], average='macro') 
        return acc, f1_macro

    def test_generalization_bp(self, graph_data, embs, label_embs, key_list):
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
        self.prepare_AGG_bp(H, dims = label_embs.shape[0])
        edge_index = graph_data.edge_index.to(self.device)
        edge_rv = edge_rv.to(self.device)
        mix_likelihood = self.model(log_probability_matrix, edge_index, edge_weight, edge_rv, deg)
        max_indices = torch.argmax(mix_likelihood, dim=1)
        mapping = {}
        key_list = sorted(key_list)
        for i in range(label_embs.shape[0]):
            mapping[i] = key_list[i]
        new_max_indices = list(map(lambda x: mapping[x], max_indices.cpu().numpy()))
        new_max_indices = torch.tensor(new_max_indices).to(self.device)
        same_flag = (labels==new_max_indices)
        test_id = graph_data.test_id
        acc = torch.sum(same_flag[test_id])/test_id.shape[0]
        f1_macro = f1_score(labels.cpu()[test_id], max_indices.cpu()[test_id], average='macro') 
        return acc, f1_macro

    def test_generalization_bp_appr(self, graph_data, embs, label_embs, key_list):
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
        self.prepare_AGG_bp_appr(weight)
        mix_likelihood = self.model(probability_matrix, edge_index)
        key_list = sorted(key_list)
        mapping = {}
        for i in range(label_embs.shape[0]):
            mapping[i] = key_list[i]
        max_indices = torch.argmax(mix_likelihood, dim=1)
        new_max_indices = list(map(lambda x: mapping[x], max_indices.cpu().numpy()))
        new_max_indices = torch.tensor(new_max_indices).to(self.device)
        same_flag = (labels==new_max_indices)
        test_id = graph_data.test_id
        acc = torch.sum(same_flag[test_id])/test_id.shape[0]
        f1_macro = f1_score(labels.cpu()[test_id], max_indices.cpu()[test_id], average='macro') 
        return acc, f1_macro

    def sort_edge(self, num_nodes, edge_index, sort_by_row=True):
        assert (edge_index.shape[1] == 0) or (0 <= edge_index.min()) and (edge_index.max() <= num_nodes-1)
        idx = edge_index[1-int(sort_by_row)]*num_nodes+edge_index[int(sort_by_row)]
        perm = idx.argsort()
        return edge_index[:, perm], perm

    def catch_answer(self, response_list, labels):
        pred_list = []
        for response in response_list:
            content = response.content
            answer = -1
            for label_idx, label in enumerate(labels):
                label = re.sub(r'\(.*?\)', '', label)
                is_present = bool(re.search(label, content, re.IGNORECASE))
                if is_present:
                    if answer == -1:
                        answer = label_idx
                    else:
                        answer = -2
                        break
            pred_list.append(answer)
        return pred_list

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

    def load_labels(self, dataset):
        label_desc = pd.read_csv(f'dataset/{dataset}/categories.csv')
        labels = []
        num_label = len(label_desc)
        num_columns = label_desc.shape[1] 
        for row in range(num_label):
            label = label_desc.iloc[row][0]
            labels.append(label)
        return labels


def main():
    parser = argparse.ArgumentParser(description='test the generalization of encoders')
    # hardware and general
    parser.add_argument('--seed', default=42)
    parser.add_argument('--device', dest = 'device', default = 0, type = int)
    parser.add_argument('--task', dest = 'task', default = 'nc', help = 'nc refers to node classification')
    
    # data and model
    parser.add_argument('--dataset', dest = 'dataset', type = str, default = 'cora', help = 'cora')
    parser.add_argument('--model', dest = 'model', type = str, default = '4o', help = 'here model refers to the LLM used to get class embeddings')
    parser.add_argument('--encoder', dest = 'encoder', type = str, default = 'sbert', help = 'encoders to select')
    args = parser.parse_args()

    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()

    