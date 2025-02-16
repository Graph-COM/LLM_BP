import argparse
import random
import importlib
from tqdm import tqdm
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch import Tensor

from torch_geometric.data import Data
from transformers import AutoTokenizer, AutoModel
import dgl


# the inference pipeline of LLM2Vec is adopted from Making text embedders few-shot learners Chaofan Li et al
# original code could be found in https://huggingface.co/BAAI/bge-en-icl

def get_new_queries(queries, query_max_len, examples_prefix, tokenizer):
    inputs = tokenizer(
        queries,
        max_length=query_max_len - len(tokenizer('<s>', add_special_tokens=False)['input_ids']) - len(
            tokenizer('\n<response></s>', add_special_tokens=False)['input_ids']),
        return_token_type_ids=False,
        truncation=True,
        return_tensors=None,
        add_special_tokens=False
    )
    prefix_ids = tokenizer(examples_prefix, add_special_tokens=False)['input_ids']
    suffix_ids = tokenizer('\n<response>', add_special_tokens=False)['input_ids']
    new_max_length = (len(prefix_ids) + len(suffix_ids) + query_max_len + 8) // 8 * 8 + 8
    new_queries = tokenizer.batch_decode(inputs['input_ids'])
    for i in range(len(new_queries)):
        new_queries[i] = examples_prefix + new_queries[i] + '\n<response>'
    return new_max_length, new_queries


def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

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
    
def load_labels(dataset):
    label_desc = pd.read_csv(f'dataset/{dataset}/categories.csv')
    labels = []
    num_label = len(label_desc)
    num_columns = label_desc.shape[1] 
    for row in range(num_label):
        label = label_desc.iloc[row][0]
        labels.append(label)
    return labels

def main():
    parser = argparse.ArgumentParser(description='generate LLM2Vec embeddings')
    parser.add_argument('--dataset', dest = 'dataset', type = str, default = 'cora', help = 'cora, pubmed, arxiv')
    parser.add_argument('--version', dest = 'version', type = str, default = 'primary', help = 'primary or class_aware')
    parser.add_argument('--split', dest = 'split', default = 'both', type = str, help = 'train or test or both')
    parser.add_argument('--seed', default=42)
    args = parser.parse_args()
    
    setup_seed(args.seed)
    
    # get the model
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-en-icl')
    model = AutoModel.from_pretrained('BAAI/bge-en-icl', device_map = 'auto')
    model.eval()
    
    # get the dataset raw text w/ label
    if args.dataset in ['cornell', 'texas', 'wisconsin', 'washington']:
        df = pd.read_csv(f'./dataset/{args.dataset}/{args.dataset.capitalize()}.csv')
        raw_texts = df.raw_text.to_list()
    else:
        raw_texts = load_raw_texts(args.dataset)
    labels = load_labels(args.dataset)

    texts_to_encode = raw_texts
    # get the task, query, examples
    module = importlib.import_module('prompts')
    Prompt = getattr(module, 'Prompt_'+args.dataset)
    prompts = Prompt(texts_to_encode, labels)
    prompts.prepare_prompts(args.version)
    task = prompts.task
    examples_prefix = prompts.examples_prefix
    queries = prompts.queries
    num_query = len(queries)

    query_max_len, doc_max_len = 4096, 4096
    new_query_max_len, new_all_text = get_new_queries(queries, query_max_len, examples_prefix, tokenizer)
    new_queries = new_all_text[:num_query]
    new_labels = new_all_text[num_query:]
    
    batch_size = 4
    print(f'Totally {len(new_queries)} sentences to encode, batch size: {batch_size}')
    for idx, batch_query in enumerate(tqdm(batch_loader(new_queries, batch_size))):
        query_batch_dict = tokenizer(batch_query, max_length=new_query_max_len, padding=True, truncation=True, return_tensors='pt')
        with torch.no_grad():
            query_outputs = model(**query_batch_dict)
            query_embeddings = last_token_pool(query_outputs.last_hidden_state, query_batch_dict['attention_mask'])
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        if idx == 0:
            all_embeddings = query_embeddings
        else: 
            all_embeddings = torch.cat((all_embeddings, query_embeddings), 0)
    torch.save(all_embeddings, f'dataset/{args.dataset}/llmicl_{args.version}_x.pt')
    

if __name__ == '__main__':
    main()