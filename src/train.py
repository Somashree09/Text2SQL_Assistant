

import argparse

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from data_utils import make_format_fn, make_tokenize_fn


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(cfg: dict):
    # -----------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------
    print(f"Loading dataset: {cfg['dataset_name']}")
    dataset = load_dataset(cfg["dataset_name"])["train"]

    dataset = dataset.shuffle(seed=cfg["seed"]).select(range(cfg["max_train_samples"]))

    split = dataset.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])
    train_ds, eval_ds = split["train"], split["test"]


    # -----------------------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    format_fn = make_format_fn(tokenizer)
    tokenize_fn = make_tokenize_fn(tokenizer, max_length=cfg["max_length"])

    train_ds = train_ds.map(format_fn, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(format_fn, remove_columns=eval_ds.column_names)

    print("\n--- Formatted training example ---\n")
    print(train_ds[0]["text"][:800])

    train_ds = train_ds.map(tokenize_fn, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tokenize_fn, remove_columns=eval_ds.column_names)

    # -----------------------------------------------------------------
    # Model + LoRA
    # -----------------------------------------------------------------
    print(f"Loading base model: {cfg['model_name']}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, device_map="auto"
    )

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg["lora_r"],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        target_modules=cfg["target_modules"],
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_train_epochs"],
        learning_rate=cfg["learning_rate"],
        logging_steps=cfg["logging_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
    )

    trainer.train()
    model.save_pretrained(cfg["adapter_dir"])
    tokenizer.save_pretrained(cfg["adapter_dir"])
    print(f"\nAdapter saved to {cfg['adapter_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
