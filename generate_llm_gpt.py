import argparse
import random
import importlib
import sys
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
from openai import OpenAI
import dgl


def setup_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False

def load_raw_texts(dataset):
    raw_texts_path = f'dataset/{dataset}/raw_texts.pt'
    raw_texts = torch.load(raw_texts_path)
    return raw_texts

def main():
    parser = argparse.ArgumentParser(description='prepare_data')
    parser.add_argument('--dataset', dest = 'dataset', type = str, default = 'cora', help = 'datasets')
    parser.add_argument('--encoder', dest = 'encoder', type = str, default = 'text-embedding-3-large', help = 'engine in gpt')
    parser.add_argument('--split', dest = 'split', default = 'both', type = str, help = 'train or test or both')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    setup_seed(args.seed)
    args.version = 'class_aware'

    # get the dataset raw text w/ label
    if args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
        df = pd.read_csv(f'./dataset/{args.dataset}/{args.dataset.capitalize()}.csv')
        raw_texts = df.raw_text.to_list()
    else:
        raw_texts = load_raw_texts(args.dataset)

    texts_to_encode = raw_texts
    # get the task, query, examples
    client = OpenAI()
    idx = 0
    for text in tqdm(texts_to_encode):
        text = text[:8000]
        response = client.embeddings.create(
            input=text,
            model='text-embedding-3-large',
        )
        emb = torch.tensor(response.data[0].embedding).reshape(1, -1)
        if idx == 0:
            all_embeddings = emb
        else:
            all_embeddings = torch.cat((all_embeddings, emb), 0)
        idx+=1
    torch.save(all_embeddings, f'dataset/{args.dataset}/llmgpt_{args.encoder}_x.pt')

if __name__ == '__main__':
    main()