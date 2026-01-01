# LLM-Document-Summarization
This repository contains an end-to-end pipeline for fine-tuning LLaMA-3-8B using QLoRA to generate concise summaries of long documents.

It includes:
1. Dataset preparation from raw text files,
2. QLoRA-based fine-tuning,
3. Evaluation using ROUGE & BERTScore.

**Environment Setup**
```
conda create --name summarization python=3.10
conda activate summarization
```
**Package Installations**
```
pip install -r requirements.txt
```
 **Llama-3-8b Model Configurations using Hugging Face**
1. Visit https://huggingface.co/meta-llama/Meta-Llama-3-8B and request access to the model.
2. Log in to Hugging Face:
 ```
huggingface-cli login
```
**Dataset**
1. Download the dataset from https://zenodo.org/records/7152317#.ZCSfaoTMI2y.
2. Run the preprocessing script to prepare the dataset for training and testing:
```
python data_pre_process.py
```
**Fine tune the model**
```

```
