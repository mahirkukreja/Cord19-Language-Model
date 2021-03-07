#!/usr/bin/env python
# coding: utf-8

# Imports

from pathlib import Path
from tokenizers import ByteLevelBPETokenizer
from tokenizers.implementations import ByteLevelBPETokenizer
from tokenizers.processors import BertProcessing
from transformers import RobertaConfig
from transformers import RobertaTokenizerFast
from transformers import RobertaForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Set Path for Training Data

paths = [str(x) for x in Path(".").glob("data/train.txt")]

# Initialize a tokenizer

tokenizer = ByteLevelBPETokenizer()

# Train the Tokenizer

tokenizer.train(files=paths, vocab_size=40000, min_frequency=10, special_tokens=[
    "<s>",
    "<pad>",
    "</s>",
    "<unk>",
    "<mask>",
])

# Make a new Directory to save model related files

!mkdir CovidBERTa
tokenizer.save_model("CovidBERTa")

# Tokenizer Processing

tokenizer = ByteLevelBPETokenizer(
    "./CovidBERTa/vocab.json",
    "./CovidBERTa/merges.txt",
)

tokenizer._tokenizer.post_processor = BertProcessing(
    ("</s>", tokenizer.token_to_id("</s>")),
    ("<s>", tokenizer.token_to_id("<s>")),
)
tokenizer.enable_truncation(max_length=512)

# Initialize config for Roberta

config = RobertaConfig(
    vocab_size=40000,
    max_position_embeddings=514,
    num_attention_heads=12,
    num_hidden_layers=6,
    type_vocab_size=1,
)

# Load Saved Tokenizer

tokenizer = RobertaTokenizerFast.from_pretrained("./CovidBERTa", max_len=512)

# Initialize Roberta for Masked Language Modelling with config initialized above.

model = RobertaForMaskedLM(config=config)

# Load Training Set

%%time
from transformers import LineByLineTextDataset

dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/train.txt",
    block_size=384,
)

# Load Validation Set

%%time
from transformers import LineByLineTextDataset

eval_dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="data/eval.txt",
    block_size=384,
)

# Setup collator for MLM

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Setup Arguments for Training based on data understanding

training_args = TrainingArguments(
    output_dir="./CovidBERTa",
    overwrite_output_dir=True,
    per_gpu_train_batch_size=16,
    save_steps=1250,
    save_total_limit=2,
    num_train_epochs=100,
    evaluation_strategy='steps',
    eval_steps=1250,
    fp16=True,
    do_train=True,
    do_eval=True,
    logging_steps=1250,
    gradient_accumulation_steps=4,
    learning_rate=1e-4,
    per_device_eval_batch_size=32,
)

# Pass all declared parameters for training to Trainer

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
)

# Train the Model

trainer.train()


