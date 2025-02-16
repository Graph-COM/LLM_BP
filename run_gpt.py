import os
import pathlib
import argparse
from tqdm import tqdm
import random
import re
import pandas as pd
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
    #torch.backends.cudnn.benchmark = False

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
        self.train_folder = './results/'+ str(self.args.task)+'_'+str(self.args.dataset)+'/'+str(self.args.model)+'/llm_baseline/'
        create_nested_folder(self.train_folder)
        # prepare data
        if self.args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
            df = pd.read_csv(f'./dataset/{self.args.dataset}/{self.args.dataset.capitalize()}.csv')
            raw_texts = df.raw_text.to_list()
        else:
            raw_texts = self.load_raw_texts(self.args.dataset)
        labels = self.load_labels(self.args.dataset)
        graph_data = self.prepare_data()
        text_to_test = [raw_texts[i] for i in graph_data.test_id]
        
        # do inference
        if self.args.mode == 'inference':
            for trial in range(1):
                print(f'Trial: {trial}')
                response_list = []
                for text in tqdm(text_to_test):
                    text = text[:10000]
                    prompt = self.prepare_prompt(labels, text)
                    response = self.generate(self.args.model, prompt)
                    print(response)
                    response_list.append(response)
                with open(self.train_folder+f'response_{trial}_{self.args.prompt}.pkl', "wb") as file:
                    pkl.dump(response_list, file)

        elif self.args.mode == 'evaluate':
            for trial in range(1):
                pth = self.train_folder+f'response_{trial}.pkl'
                with open(pth, "rb") as file:
                    loaded_list = pkl.load(file)
                pred_list = self.catch_answer(loaded_list, labels)
                pred_list = np.array(pred_list)
                gt = graph_data.y[graph_data.test_id]
                mask = pred_list==gt.numpy()
                f1_macro = f1_score(pred_list, gt.numpy(), average='macro')
                print(f'Accuracy: {np.sum(mask)/pred_list.shape[0]}')
                print(f'F1: {f1_macro}')


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
        

    def generate(self, model, prompt):
        client = OpenAI()
        if model == '4o_mini':
            engine = "gpt-4o-mini"
        elif model == '4o':
            engine = 'gpt-4o'
        elif model == '35_t':
            engine = 'gpt-3.5-turbo'
        else:
            raise NotImplementedError
        completion = client.chat.completions.create(
            model=engine,
            messages=prompt
        )
        response = completion.choices[0].message
        return response
    
    def prepare_prompt(self, labels, content):
        init_instruct_1 = f'We have {prompt_dict[self.args.dataset]} from the following {self.num_class} categories: {labels}'
        init_instruct_2 = f'The text is as follows:'
        init_instruct_3 = f'Please tell which category the text belongs to:'
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
        if self.args.dataset in ['wikics']:
            pkg = torch.where(graph_data.test_mask)
            pkg2 = torch.where(graph_data.train_mask)
            graph_data.test_id = pkg[0]
            graph_data.train_id = pkg2[0]
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
    parser.add_argument('--model', dest = 'model', type = str, default = '4o_mini', help = '4o_mini, 4o, 35_t')
    parser.add_argument('--mode', dest = 'mode', type = str, default = 'inference', help = 'inference or evaluate')
    args = parser.parse_args()

    setup_seed(args.seed)

    runner = Runner(args)
    runner.run()


if __name__ == '__main__':
    main()

    