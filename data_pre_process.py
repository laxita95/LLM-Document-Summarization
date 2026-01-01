import os
import json
import random
from tqdm import tqdm

# -----------------------------
# PATHS
# -----------------------------
judgement_dir = "../dataset/IN-Ext/judgement/"
full_summary_dir = "../dataset/IN-Ext/summary/full/"
segment_summary_dir = "../dataset/IN-Ext/summary/segment-wise/"
output_dir = "../processed-IN-Ext/"

os.makedirs(output_dir, exist_ok=True)

# -----------------------------
# TRAIN / TEST SPLIT
# -----------------------------
def train_test_split(data, train_ratio=0.7, seed=42):
    random.seed(seed)
    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    return data[:split_idx], data[split_idx:]

# -----------------------------
# SAVE JSONL
# -----------------------------
def save_jsonl(data, path):
    with open(path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

# -----------------------------
# LOAD FULL SUMMARIES
# -----------------------------
def load_full_summaries(judgement_dir, full_summary_dir, author):
    data = []
    for filename in tqdm(os.listdir(judgement_dir), desc=f"Full summaries {author}"):
        if filename.endswith(".txt"):
            judgement_path = os.path.join(judgement_dir, filename)
            summary_path = os.path.join(full_summary_dir, author, filename)

            if os.path.exists(summary_path):
                with open(judgement_path, "r", encoding="utf-8") as f:
                    judgement = f.read()
                with open(summary_path, "r", encoding="utf-8") as f:
                    summary = f.read()

                data.append({
                    "filename": filename,
                    "judgement": judgement,
                    "summary": summary,
                    "author": author
                })
    return data

# -----------------------------
# LOAD SEGMENT SUMMARIES
# -----------------------------
def load_segment_summaries(segment_summary_dir, author):
    data = []
    segments = ["analysis", "argument", "facts", "judgement", "statute"]
    base_path = os.path.join(segment_summary_dir, author, "analysis")

    for filename in tqdm(os.listdir(base_path), desc=f"Segment summaries {author}"):
        if filename.endswith(".txt"):
            segment_text = {}
            for segment in segments:
                segment_path = os.path.join(segment_summary_dir, author, segment, filename)
                if os.path.exists(segment_path):
                    try:
                        with open(segment_path, "r", encoding="utf-8") as f:
                            segment_text[segment] = f.read()
                    except UnicodeDecodeError:
                        with open(segment_path, "r", encoding="latin-1") as f:
                            segment_text[segment] = f.read()

            data.append({
                "filename": filename,
                "segments": segment_text,
                "author": author
            })
    return data

# -----------------------------
# LOAD DATA
# -----------------------------
print("\nLoading full summaries...")
full_A1 = load_full_summaries(judgement_dir, full_summary_dir, "A1")
full_A2 = load_full_summaries(judgement_dir, full_summary_dir, "A2")

print("\nLoading segment-wise summaries...")
segment_A1 = load_segment_summaries(segment_summary_dir, "A1")
segment_A2 = load_segment_summaries(segment_summary_dir, "A2")

# -----------------------------
# SPLIT DATA
# -----------------------------
train_full_A1, test_full_A1 = train_test_split(full_A1)
train_full_A2, test_full_A2 = train_test_split(full_A2)

train_segment_A1, test_segment_A1 = train_test_split(segment_A1)
train_segment_A2, test_segment_A2 = train_test_split(segment_A2)

# -----------------------------
# SAVE DATASETS
# -----------------------------
print("Saving datasets...")

save_jsonl(train_full_A1, os.path.join(output_dir, "train_full_A1.jsonl"))
save_jsonl(test_full_A1,  os.path.join(output_dir, "test_full_A1.jsonl"))

save_jsonl(train_full_A2, os.path.join(output_dir, "train_full_A2.jsonl"))
save_jsonl(test_full_A2,  os.path.join(output_dir, "test_full_A2.jsonl"))

save_jsonl(train_segment_A1, os.path.join(output_dir, "train_segment_A1.jsonl"))
save_jsonl(test_segment_A1,  os.path.join(output_dir, "test_segment_A1.jsonl"))

save_jsonl(train_segment_A2, os.path.join(output_dir, "train_segment_A2.jsonl"))
save_jsonl(test_segment_A2,  os.path.join(output_dir, "test_segment_A2.jsonl"))

print("Dataset preparation complete!")
print("Output directory:", output_dir)
