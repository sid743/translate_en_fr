import os
from dataclasses import dataclass
from typing import Dict, List, Any

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    Trainer,
)
import transformers


# ========= CONFIG =========
MODEL_NAME = "t5-small"              # you can later try "t5-base"
TSV_PATH = "en_fr_synthetic.tsv"     # change if your file has a different name
MAX_SOURCE_LENGTH = 64
MAX_TARGET_LENGTH = 64
VAL_SPLIT = 0.1                      # 10% validation
SEED = 42
OUTPUT_DIR = "./mt_en_fr_t5"
FINAL_DIR = "./mt_en_fr_t5_final"
# ==========================


def main():
    print("Transformers version at runtime:", transformers.__version__)
    print("Transformers module file:", transformers.__file__)
    print("Using TSV file:", TSV_PATH)

    # 1. Load TSV as a raw dataset (english<TAB>french)
    raw_dataset = load_dataset(
        "csv",
        data_files={"data": TSV_PATH},
        delimiter="\t",
        column_names=["en", "fr"],
    )["data"]

    # 2. Make it 2-way: En->Fr and Fr->En
    def make_two_way(examples):
        inputs = []
        targets = []
        for en, fr in zip(examples["en"], examples["fr"]):
            # en -> fr
            inputs.append(f"translate English to French: {en}")
            targets.append(fr)
            # fr -> en
            inputs.append(f"translate French to English: {fr}")
            targets.append(en)
        return {"input_text": inputs, "target_text": targets}

    two_way_dataset = raw_dataset.map(
        make_two_way,
        batched=True,
        remove_columns=["en", "fr"],
    )

    # 3. Train/validation split
    two_way_dataset = two_way_dataset.shuffle(seed=SEED)
    split = two_way_dataset.train_test_split(test_size=VAL_SPLIT, seed=SEED)
    train_dataset = split["train"]
    val_dataset = split["test"]

    print("Example after 2-way expansion:")
    print(train_dataset[0])

    # 4. Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    # 5. Tokenization function (no as_target_tokenizer)
    def preprocess(examples):
        # Inputs
        model_inputs = tokenizer(
            examples["input_text"],
            max_length=MAX_SOURCE_LENGTH,
            truncation=True,
        )

        # Targets
        labels = tokenizer(
            text_target=examples["target_text"],
            max_length=MAX_TARGET_LENGTH,
            truncation=True,
        )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized_train = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["input_text", "target_text"],
    )
    tokenized_val = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=["input_text", "target_text"],
    )

    # 6. Data collator
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    # 7. Training arguments (simple, compatible)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=100,
        learning_rate=3e-4,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,  # effective batch size 32
        weight_decay=0.01,
        save_total_limit=2,
        num_train_epochs=3,
        fp16=torch.cuda.is_available(),  # use mixed precision if GPU supports it
    )

    # 8. Trainer (no fancy seq2seq extras, we just train on loss)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 9. Train!
    trainer.train()

    # 10. Save final model
    os.makedirs(FINAL_DIR, exist_ok=True)
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    print("Final model saved to:", FINAL_DIR)


if __name__ == "__main__":
    main()
