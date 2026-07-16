"""
Quick CLI for querying the fine-tuned model.

Usage:
    python src/inference.py --config configs/config.yaml \
        --schema "CREATE TABLE users (id INT, name TEXT, age INT)" \
        --question "How many users are older than 30?"
"""

import argparse

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import build_prompt


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--schema", type=str, required=True)
    parser.add_argument("--question", type=str, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, cfg["adapter_dir"])
    model.eval()

    prompt = build_prompt(args.schema, args.question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=cfg["max_new_tokens"],
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    sql = full_text.split("### SQL:")[-1].strip()
    print(sql)


if __name__ == "__main__":
    main()
