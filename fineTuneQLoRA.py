import os, json, torch
from datasets import Dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model

# -------------------------
# ENV (CRITICAL)
# -------------------------
os.environ["ACCELERATE_DISABLE_BF16"] = "1"
os.environ["ACCELERATE_USE_AMP"] = "false"

# -------------------------
# CONFIG
# -------------------------
MODEL_NAME = "meta-llama/Meta-Llama-3-8B"
DATA_DIR = "/home/gpuuser6/Alik/Laxita/Updated_Code/dataset/processed-IN-Ext/"
OUTPUT_DIR = "./llama3_legal_qlora_70"

# -------------------------
# TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

# LOCK PAD TOKEN
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# QLORA CONFIG
# -------------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# -------------------------
# MODEL
# -------------------------
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)

# LOCK PAD TOKEN IN MODEL
model.config.pad_token_id = tokenizer.pad_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

# Gradient checkpointing (safe)
model.gradient_checkpointing_enable()

# -------------------------
# LoRA CONFIG
# -------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# -------------------------
# DATASET
# -------------------------
def load_dataset(jsonl_file):
    with open(jsonl_file, encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    texts = []
    for item in data:
        texts.append(
            f"""### Instruction:
Summarize the following legal judgement clearly and concisely.

### Input:
{item['judgement'][:12000]}

### Response:
{item['summary']}"""
        )
    return Dataset.from_dict({"text": texts})

train_data = concatenate_datasets([
    load_dataset(os.path.join(DATA_DIR, "train_full_A1.jsonl")),
    load_dataset(os.path.join(DATA_DIR, "train_full_A2.jsonl")),
])

# -------------------------
# TRAINING CONFIG (NO AMP)
# -------------------------
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,

    # AMP OFF (THIS FIXES THE CRASH)
    fp16=False,
    bf16=False,

    max_length=4096,
    logging_steps=10,
    save_strategy="epoch",
    report_to="none",

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# -------------------------
# TRAINER
# -------------------------
trainer = SFTTrainer(
    model=model,
    train_dataset=train_data,
    peft_config=lora_config,
    args=training_args,
)

trainer.train()

model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
