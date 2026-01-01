import torch
import json
import evaluate
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "./llama3_legal_qlora_70"
TEST_JSONL = "/home/gpuuser6/Alik/Laxita/Updated_Code/dataset/processed-IN-Ext/test_full_A2.jsonl"
MAX_INPUT_TOKENS = 4096
MAX_NEW_TOKENS = 500
MAX_SAMPLES = 100  # set None for full dataset

# -------------------------
# LOAD METRICS
# -------------------------
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# MODEL (QLORA SAFE)
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="auto",
    quantization_config=bnb_config,
)

model.eval()

# -------------------------
# LOAD TEST DATA
# -------------------------
def load_test_data(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "judgement": item["judgement"],
                "reference": item["summary"]
            })
    return Dataset.from_list(data)

dataset = load_test_data(TEST_JSONL)

if MAX_SAMPLES:
    dataset = dataset.select(range(min(MAX_SAMPLES, len(dataset))))

# -------------------------
# GENERATE + COLLECT
# -------------------------
predictions = []
references = []

for sample in tqdm(dataset):
    prompt = f"""### Instruction:
Summarize the following legal judgement clearly and concisely.

### Input:
{sample['judgement'][:12000]}

### Response:
"""

    inputs = tokenizer(
        prompt,
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    pred = tokenizer.decode(
        output_ids[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )

    predictions.append(pred.strip())
    references.append(sample["reference"].strip())

# -------------------------
# COMPUTE METRICS
# -------------------------
rouge_scores = rouge.compute(
    predictions=predictions,
    references=references
)

bert_scores = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)

results = {
    "ROUGE-1": round(rouge_scores["rouge1"], 4),
    "ROUGE-2": round(rouge_scores["rouge2"], 4),
    "ROUGE-L": round(rouge_scores["rougeL"], 4),
    "BERTScore-F1": round(sum(bert_scores["f1"]) / len(bert_scores["f1"]), 4),
}

print("\n========== EVALUATION RESULTS ==========")
for k, v in results.items():
    print(f"{k}: {v}")
