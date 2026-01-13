#!/usr/bin/env python3
"""
Train a small GPT model on DeepAll data using HuggingFace Transformers
Based on Andrej Karpathy's approach: minimal, hackable, educational
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    AutoTokenizer,
)

print("=" * 70)
print("DeepAll Training with HuggingFace Transformers")
print("=" * 70)
print()

# Setup
device = "cpu"
REPO_ROOT = Path(__file__).resolve().parents[1]
data_dir = REPO_ROOT / "deepall" / "nanogpt-pytorch-deepall-v1" / "data jsonl"
output_dir = Path.home() / ".cache" / "nanochat" / "deepall_model"
output_dir.mkdir(parents=True, exist_ok=True)

# Step 1: Load data
print("Step 1: Loading DeepAll data...")
all_texts = []
for jsonl_file in data_dir.glob("*.jsonl"):
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data = json.loads(line)
                prompt = data.get('prompt', '')
                completion = data.get('completion', '')
                combined = prompt + completion
                all_texts.append(combined)
            except:
                pass

print(f"✓ Loaded {len(all_texts)} examples")
print()

# Step 2: Create tokenizer
print("Step 2: Creating tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print(f"✓ Tokenizer ready (vocab_size: {tokenizer.vocab_size})")
print()

# Step 3: Tokenize data
print("Step 3: Tokenizing data...")
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )

dataset = Dataset.from_dict({"text": all_texts})
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

print(f"✓ Tokenized {len(tokenized_dataset)} examples")
print()

# Step 4: Create model
print("Step 4: Creating model...")
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=256,
    n_embd=256,
    n_layer=4,
    n_head=4,
)
model = GPT2LMHeadModel(config)
print(f"✓ Model created")
print(f"  Parameters: {model.num_parameters():,}")
print()

# Step 5: Train
print("Step 5: Training...")
training_args = TrainingArguments(
    output_dir=str(output_dir),
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=2,
    save_steps=10,
    save_total_limit=2,
    logging_steps=5,
    learning_rate=5e-5,
    weight_decay=0.01,
    warmup_steps=10,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

trainer.train()
print()
print("✓ Training complete!")
print()

# Step 6: Save model
print("Step 6: Saving model...")
model.save_pretrained(str(output_dir))
tokenizer.save_pretrained(str(output_dir))
print(f"✓ Model saved to {output_dir}")
print()

print("=" * 70)
print("✓ Done!")
print("=" * 70)
print()
print("Your model is ready!")
print(f"Location: {output_dir}")
print()

