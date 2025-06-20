# LLM_BP

[[Paper]]([https://arxiv.org/abs/2410.20724](https://arxiv.org/html/2502.11836v2))

![model](framework_241104.png)

## Overview
This is the official implementation of paper ['Model Generalization on Text Attribute Graphs: Principles with Large Language Models'](http://arxiv.org/abs/2502.11836), Haoyu Wang, Shikun Liu, Rongzhe Wei, Pan Li.

## Repository Structure

The repository structure is as follows:

```
LLM_BP (Root Directory)
│── dataset/        # Contains dataset files
│── model/          # Stores model implementation of LLM-BP and LLM-BP (appr.)
│── results/        # Contains generated results from GPT-4o (the predictions on testset) and GPT-4o-mini (predictions on homophily ratio)
│── zero_shot.py  # zero shot inference
│── few_shot.py        # few shot inference
│── run_gpt.py     # run openai GPT to predict the results by taking raw node texts
│── pred_h.py     # predict the homophily ratio r by sampling edges
│── generate_llm.py     # generate the embeddings of vanilla LLM2Vec or task-adaptive encoder
│── generate_lm.py     # generate the embeddings of sbert or Roberta
│── generate_llm_gpt.py     # generate the embeddings of text-embedding-3-large
│── README.md       # Documentation file
```

## STEP 0.1 Environment Setup

To set up the environment, follow these steps:

```
conda create -n llmbp python==3.8.18 
conda activate llmbp
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install pyg_lib==0.3.1+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_scatter==2.1.2 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_sparse==0.6.18+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_cluster==1.6.3+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html 
pip install torch_spline_conv==1.2.2+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install transformers==4.46.3 
pip install sentence_transformers==2.2.2
pip install dgl==2.4.0+cu121 -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html 
pip install openai 
pip install torch_geometric==2.5.0 
pip install protobuf 
pip install accelerate
```
-------------

## STEP 0.2 Dataset Preparation

The dataset structure should be organized as follows:

```plaintext
/dataset/
│── [dataset_name]/
│   │── processed_data.pt    # Contains labels and graph information
│   │── [encoder]_x.pt       # Features extracted by different encoders
│   │── categories.csv       # label name raw texts
│   │── raw_texts.pt       # raw text of each node
```

### File Descriptions
- **`processed_data.pt`**: A PyTorch file storing the processed dataset, including graph structure and node labels. Note that in heterophilic datasets, thie is named as [Dataset].pt, where Dataset could be Cornell, etc, and should be opened with DGL.
- **`[encoder]_x.pt`**: Feature matrices extracted using different encoders, where `[encoder]` represents the encoder name.
- **`categories.csv`**: raw label names.
- **`raw_texts.pt`**: raw node texts. Note that in heterophilic datasets, this is named as [Dataset].csv, where Dataset can be Cornell, etc.

### Dataset Naming Convention
`[dataset_name]` should be one of the following:
- `cora`
- `citeseer`
- `pubmed`
- `bookhis`
- `bookchild`
- `sportsfit`
- `wikics`
- `cornell`
- `texas`
- `wisconsin`
- `washington`

### Encoder Naming Convention
`[encoder]` can be one of the following:
- `sbert` (the sentence-bert encoder)
- `roberta` (the Roberta encoder)
- `llmicl_primary` (the vanilla LLM2Vec)
- `llmicl_class_aware` (the task-adaptive encoder)
- `llmgpt_text-embedding-3-large` (the embedding api text-embedding-3-large by openai)


Ensure the datasets are placed correctly for smooth execution.

### Download Pre-Calculated Embeddings and datasets

They could be found at: [huggingface repository](https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs), one could directly download, place under /dataset/ folder.

## STEP 1: Generating Dataset Embeddings

```
python generate_llm.py --dataset [DATASET] --version [VERSION]
```

### Example: 
```
CUDA_VISIBLE_DEVICES=0,1 python generate_llm.py --dataset cora --version class_aware
```

### Parameters:
- `[DATASET]`: The name of the dataset.
- `[VERSION]`: 
  - `primary` → Vanilla LLM2Vec
  - `class_aware` → Task-adaptive encoding

Ensure that the appropriate CUDA devices are set before running the script.

### Download pre-calculated embeddings

We have enclosed the pre-calculated embeddings for the encoders in: [huggingface repository](https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs), one may directly download and put them under the /dataset folder

------------

## STEP 2: Generate the predictions from GPT-4o

```
python run_gpt.py --mode [MODE] --model [MODEL] --dataset [DATASET]
```

### Parameters:
- `[MODEL]`: The model selection (e.g., 4o for GPT-4o).
- `[DATASET]`: The name of the dataset.
- `[MODE]`: when set as inference, it do inference and save results, when set as evaluate, it evaluate the results of the model



### Download pre-calculated predictions

We have enclosed the pre-calculated predictions from GPT-4o in: [huggingface repository](https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs), one may directly download and put them under the /results folder

----------------

## STEP 3: Predict the homophily ratio  of the dataset

```
python pred_r.py --mode [MODE] --dataset [DATASET] --model [MODEL]
```
### Parameters:
- `[DATASET]`: The name of the dataset.
- `[MODEL]`: The model selection (e.g., 4o_mini).
- `[MODE]`: when set as inference, it do inference and save results, when set as evaluate, it makes prediction with the model

### Fill the value
Fill the predicted value in H_dict in zero_shot.py or few_shot.py

### Download pre-calculated predictions

We have enclosed the pre-calculated predictions from GPT-4o-mini in: [huggingface repository](https://huggingface.co/datasets/Graph-COM/Text-Attributed-Graphs), one may directly download and put them under the /results folder

## STEP 4: Zero-shot Inference

```
python zero_shot.py --dataset [DATASET] --encoder [ENCODER] --model 4o
```

### Parameters:
- `[DATASET]`: The name of the dataset.
- `[ENCODER]`: The encoder model (e.g., sbert, roberta, llmicl_primary, llmicl_class_aware, llmicl_text-embedding-3-large, etc.).
- `4o`: Specifies the use of GPT-4o as averaged class embeddings.

## STEP 5: Few-shot Inference

```
python few_shot.py --dataset [DATASET] --encoder [ENCODER]
```

### Parameters:
- `[DATASET]`: The name of the dataset.
- `[ENCODER]`: The encoder model (e.g., sbert, roberta, llmicl_primary, llmicl_class_aware, llmicl_text-embedding-3-large, etc.).


## Acknowledgements
The dataset pre-processing, formats and code implementations are inspired by or built upon [GLBench](https://github.com/NineAbyss/GLBench), [Text-space graph foundation model](https://github.com/CurryTang/TSGFM), and [LLaGA](https://github.com/VITA-Group/LLaGA).


## Citation

If you find our work helpful, please consider citing:

```
@article{wang2025model,
title={Model Generalization on Text Attribute Graphs: Principles with Large Language Models},
author={Wang, Haoyu and Liu, Shikun and Wei, Rongzhe and Li, Pan},
journal={arXiv preprint arXiv:2502.11836},
year={2025}
}
```
