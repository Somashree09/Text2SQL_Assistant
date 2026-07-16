"""
Evaluation script for the fine-tuned text-to-SQL model.

Usage:
    python src/evaluate.py --config configs/config.yaml
"""

import argparse
import csv
import re

import torch
import yaml
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_utils import build_prompt
from metrics import exact_match, execution_metrics, component_match


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def generate_sql(model, tokenizer, context, question, max_new_tokens):
    prompt = build_prompt(context, question)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )
    full_text = tokenizer.decode(output[0], skip_special_tokens=True)
    generated = full_text.split("### SQL:")[-1].strip()
    generated = re.split(r"\n\s*###|\n\n", generated)[0].strip()
    return generated


def main(cfg: dict):
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        cfg["model_name"], torch_dtype=torch.bfloat16, device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, cfg["adapter_dir"])
    model.eval()

    print(f"Loading dataset: {cfg['dataset_name']}")
    dataset = load_dataset(cfg["dataset_name"])["train"]
    dataset = dataset.train_test_split(test_size=cfg["test_size"], seed=cfg["seed"])["test"]
    if cfg.get("num_eval_samples"):
        dataset = dataset.select(range(min(cfg["num_eval_samples"], len(dataset))))

    results = []
    print(f"Evaluating on {len(dataset)} examples...\n")

    for i, row in enumerate(dataset):
        context, question, gold_sql = row["context"], row["question"], row["answer"]
        pred_sql = generate_sql(model, tokenizer, context, question, cfg["max_new_tokens"])

        em = exact_match(pred_sql, gold_sql)
        valid, ex_match = execution_metrics(context, pred_sql, gold_sql, cfg["rows_per_table"])
        comp = component_match(pred_sql, gold_sql)

        results.append({
            "question": question,
            "gold": gold_sql,
            "pred": pred_sql,
            "exact_match": em,
            "valid_execution": valid,
            "execution_match": ex_match,
            "component_match": round(comp, 3),
        })

        if (i + 1) % 20 == 0:
            print(f"  ...{i + 1}/{len(dataset)} done")

    n = len(results)
    em_rate = sum(r["exact_match"] for r in results) / n
    valid_rate = sum(r["valid_execution"] for r in results) / n
    ex_rate = sum(r["execution_match"] for r in results) / n
    comp_rate = sum(r["component_match"] for r in results) / n

    print("\n" + "=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"Samples evaluated:        {n}")
    print(f"Exact Match (EM):         {em_rate:.2%}")
    print(f"Valid SQL rate:           {valid_rate:.2%}")
    print(f"Execution Accuracy (EX):  {ex_rate:.2%}   <-- primary metric")
    print(f"Component Match (avg):    {comp_rate:.2%}")
    print("=" * 55)

    print("\n--- Sample failures (execution mismatch) ---\n")
    for f in [r for r in results if not r["execution_match"]][:5]:
        print(f"Q: {f['question']}")
        print(f"  Gold: {f['gold']}")
        print(f"  Pred: {f['pred']}")
        print(f"  Valid execution: {f['valid_execution']}\n")

    out_path = "eval_results.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"Full results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    args = parser.parse_args()
    main(load_config(args.config))
