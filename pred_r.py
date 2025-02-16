import os
import pandas as pd
import argparse
from tqdm import tqdm
import random
import re
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

prompt_dict = {
    'cora': 'opening text of machine learning papers',
    'citeseer': 'description or opening text of scientific publications',
    'pubmed': 'title and abstract of scientific publications',
    'ogbn-arxiv': 'description or opening text of scientific publications',
    'wikics': 'entry and content of wikipedia',
    'bookhis': 'description or title of the book',
    'bookchild': 'description or title of the child literature',
    'sportsfit': 'the title of a good in sports & fitness',
    'cornell': 'webpage text',
    'texas': 'webpage text',
    'wisconsin': 'webpage text',
    'washington': 'webpage text',
}

def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def create_nested_folder(path):
    os.makedirs(path, exist_ok=True)
        
class Runner():
    def __init__(self, args):
        self.args = args
        self.num_class_dict = {'cora': 7, 'citeseer': 6, 'pubmed': 3, 
                                'bookhis': 12, 'bookchild': 24, 'sportsfit': 13, 'wikics': 10, 
                                'cornell': 5, 'texas': 5, 'washington': 5, 'wisconsin': 5}
        self.num_class = self.num_class_dict[self.args.dataset]
            
    def run(self):
        self.train_folder = './results/'+ str(self.args.task)+'_'+str(self.args.dataset)+'/'+str(self.args.model)+'/agenth/'
        create_nested_folder(self.train_folder)
        
        # prepare data
        if self.args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
            df = pd.read_csv(f'./dataset/{self.args.dataset}/{self.args.dataset.capitalize()}.csv')
            raw_texts = df.raw_text.to_list()
        else:
            raw_texts = self.load_raw_texts(self.args.dataset)
        labels = self.load_labels(self.args.dataset)
        graph_data = self.prepare_data()
        edge_index = graph_data.edge_index

        # do inference
        if self.args.mode == 'inference':
            for trial in range(5):
                print(f'Trial: {trial}')
                response_list = []
                epoch = self.args.n_total // self.args.batch_size
                for epoch_idx in tqdm(range(epoch)):
                    content = self.sample_content(raw_texts, edge_index, self.args.batch_size)
                    prompt = self.prepare_prompt(labels, content, self.args.batch_size)
                    response = self.generate(self.args.model, prompt)
                    print(response)
                    response_list.append(response)
                with open(self.train_folder+f'response_{trial}.pkl', "wb") as file:
                    pkl.dump(response_list, file)

        # do evaluation
        elif self.args.mode == 'evaluate':
            preds = np.zeros(self.args.n_total // self.args.batch_size)
            for trial in range(5):
                with open(self.train_folder+f'response_{trial}.pkl', "rb") as file:
                    loaded_list = pkl.load(file)
                pred_list = self.catch_answer(loaded_list)
                preds += np.array(pred_list)
            mask = preds>=3
            if self.args.dataset in ['cora', 'cornell', 'texas', 'wisconsin', 'washington']:
                mask = mask[:50]
                preds = preds[:50]
            print(f'predicted r: {np.sum(mask)/preds.shape[0]}')

    def catch_answer(self, response_list):
        pred_list = []
        for response in response_list:
            if self.args.model in ['4o', '4o_mini', '35t']:
                content = response.content
            else:
                content = response
            if self.args.model in ['4o', '4o_mini', '35t']:
                if '**not**' in content or 'not belong' in content or 'different categories' in content or 'No' in content:
                    pred_list.append(0)
                else:
                    pred_list.append(1)
        return pred_list
        

    def generate(self, model, prompt):
        if model in ['4o_mini', '4o', '35t']:
            client = OpenAI()
            if model == '4o_mini':
                engine = "gpt-4o-mini"
            elif model == '4o':
                engine = 'gpt-4o'
            elif model == '35t':
                engine = 'gpt-3.5-turbo'
            else:
                raise NotImplementedError
            completion = client.chat.completions.create(
                model=engine,
                messages=prompt
            )
            response = completion.choices[0].message
        return response

    def sample_content(self, raw_texts, edge_index, batch_size):
        edge_ids = torch.randint(0, edge_index.shape[1], (batch_size,))
        txt = ''
        for edge_id in edge_ids:
            node_1 = edge_index[0][edge_id].item()
            node_2 = edge_index[1][edge_id].item()
            text_1 = raw_texts[node_1]
            text_2 = raw_texts[node_2]
            pair = '[ The first text is: '+text_1+' \n The second text is: '+text_2+']'
            txt+=pair
            txt+='\n'
        return txt

    
    def prepare_prompt(self, labels, content, batch_size):
        init_instruct_1 = f'We have two {prompt_dict[self.args.dataset]} from the following {self.num_class} categories: {labels}'
        init_instruct_2 = f'The texts are as follows:'
        init_instruct_3 = f'Please tell whether they belong to the same category or not by answering Yes or No after reasoning step by step'
        messages = [
            {"role": "system",
            "content": "You are a chatbot who is an expert in text classification",},
            {"role": "user", "content": init_instruct_1+'\n'+init_instruct_2+'\n'+content+'\n'+init_instruct_3},
        ]
        return messages
        
    
    def load_raw_texts(self, dataset):
        raw_texts_path = f'dataset/{dataset}/raw_texts.pt'
        raw_texts = torch.load(raw_texts_path)
        return raw_texts

    def prepare_data(self):
        if self.args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
            dgl_graph = dgl.load_graphs(f'./dataset/{self.args.dataset}/{self.args.dataset.capitalize()}.pt')[0][0]
            edge_index = torch.stack(dgl_graph.edges())
            graph_data = Data(edge_index = edge_index, y = dgl_graph.ndata['label'])
            graph_data.test_id = torch.arange(len(graph_data.y))
            graph_data.train_id = torch.arange(len(graph_data.y))
        else:
            graph_data = torch.load(f'./dataset/{self.args.dataset}/processed_data.pt')
        return graph_data

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
    parser = argparse.ArgumentParser(description='agent to get H')
    # hardware and general
    parser.add_argument('--seed', default=42)
    # path
    parser.add_argument('--task', dest = 'task', default = 'nc', help = 'nc')
    # data
    parser.add_argument('--dataset', dest = 'dataset', type = str, default = 'cora', help = 'cora')
    parser.add_argument('--model', dest = 'model', type = str, default = '4o_mini', help = 'the model to predict h')
    parser.add_argument('--n_total', dest = 'n_total', type = int, default = 100)
    parser.add_argument('--batch_size', dest = 'batch_size', type = int, default = 1)
    parser.add_argument('--mode', dest = 'mode', type = str, default = 'inference', help = 'inference or evaluate')
    args = parser.parse_args()

    setup_seed(args.seed)

    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()

    