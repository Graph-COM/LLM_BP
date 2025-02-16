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

from transformers import AutoTokenizer, AutoModel

from sentence_transformers import SentenceTransformer


def batch_loader(data_list, batch_size):
    for i in range(0, len(data_list), batch_size):
        yield data_list[i:i + batch_size]


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
    parser.add_argument('--encoder', dest = 'encoder', type = str, default = 'sbert', help = 'roberta, sbert')
    parser.add_argument('--d', dest = 'd', type = str, default = 0, help = 'device')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    
    setup_seed(args.seed)
    
    # get the model
    device = torch.device('cuda:'+str(args.d))
    if args.encoder.startswith('sbert'):
        model = SentenceTransformer('all-MiniLM-L6-v2', device = device)
    elif args.encoder.startswith('roberta'):
        model = SentenceTransformer('all-roberta-large-v1', device = device)
    
    if args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
        df = pd.read_csv(f'./dataset/{args.dataset}/{args.dataset.capitalize()}.csv')
        raw_texts = df.raw_text.to_list()
    else:
        raw_texts = load_raw_texts(args.dataset)

    embeds = model.encode(raw_texts, batch_size=8, show_progress_bar=True)
    emb_path = f'dataset/{args.dataset}/{args.encoder}_x.pt'
    torch.save(torch.tensor(embeds), emb_path)
    
    

if __name__ == '__main__':
    main()