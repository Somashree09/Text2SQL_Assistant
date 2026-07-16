# 🚀 Text2SQL using Qwen3 + LoRA

> Fine-tuning **Qwen3-0.6B-Base** using **LoRA** to translate natural language questions into executable SQL queries.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![Transformers](https://img.shields.io/badge/🤗-Transformers-yellow)
![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)


---

## Overview

This project demonstrates an end-to-end **Text-to-SQL** system built using **Qwen3-0.6B-Base** and **LoRA (Low-Rank Adaptation)**.

The repository contains:

- Training Pipeline
- Inference Pipeline
- Evaluation Pipeline
- LoRA Adapter Saving
- SQL Quality Metrics

---

# Architecture
                    ┌────────────────────┐
                    │ SQL Dataset        │
                    └─────────┬──────────┘
                              │
                              ▼
                    Prompt Construction
                              │
                              ▼
                 Qwen3-0.6B-Base + LoRA
                              │
                ┌─────────────┴─────────────┐
                │                           │
                ▼                           ▼
         Save Adapter                 Generate SQL
                                              │
                                              ▼
                                   Evaluation Pipeline

# Project Workflow
Natural Language Question
            │
            ▼
+---------------------------+
| Prompt Formatting         |
+---------------------------+
            │
            ▼
+---------------------------+
| Fine-tuned Qwen3 Model    |
+---------------------------+
            │
            ▼
Generated SQL Query
            │
            ▼
Evaluation Metrics
# Training Pipeline
Dataset
   │
   ▼
Load Dataset
   │
   ▼
Format Prompt
   │
   ▼
Tokenization
   │
   ▼
Load Base Model
   │
   ▼
Apply LoRA
   │
   ▼
Fine-tuning
   │
   ▼
Save Adapter

# Evaluation Pipeline

                    Predicted SQL
                          │
                          ▼
             +-------------------------+
             | SQL Normalization       |
             +-------------------------+
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
     Exact Match      Valid SQL      Clause Match
                          │
                          ▼
             Synthetic Execution
                    Accuracy
                          │
                          ▼
              Evaluation Report (.csv)


# Repository Structure
text2sql-qwen-lora/

│── configs/
│     └── config.yaml
│
│── outputs/
│     ├── adapter/
│     └── evaluation/
│
│── src/
│     ├── data_utils.py
│     ├── train.py
│     ├── inference.py
│     ├── evaluate.py
│     └── metrics.py
│
│── requirements.txt
│── README.md


# Model

| Component   | Value              |
| ----------- | ------------------ |
| Base Model  | Qwen3-0.6B-Base    |
| Fine-tuning | LoRA               |
| Dataset     | SQL Create Context |
| Framework   | Hugging Face       |
| GPU         | Google Colab T4    |


# Evaluation Results

| Metric                       | Score      |
| ---------------------------- | ---------- |
| Exact Match                  | **78.50%** |
| Valid SQL                    | **98.00%** |
| Synthetic Execution Accuracy | **93.50%** |
| Clause Match                 | **90.97%** |


# Example

Which department has the highest salary?

SELECT department
FROM employee
ORDER BY salary DESC
LIMIT 1;

# Installation
git clone https://github.com/yourname/text2sql-qwen-lora.git

cd text2sql-qwen-lora

pip install -r requirements.txt

# Training
python src/train.py --config configs/config.yaml

# Evaluation
python src/evaluate.py --config configs/config.yaml

# Inference
python src/inference.py --config configs/config.yaml

# Current Limitations
-Trained on a subset (5,000 samples)
-Synthetic execution evaluation (not official Spider execution)
-Single-model support

# Future Improvements
-Official Spider evaluation
-FastAPI backend
-Streamlit frontend
-Docker support
-TensorBoard logging
-Hugging Face Spaces deployment

# Tech Stack
-Python
-PyTorch
-Hugging Face Transformers
-PEFT (LoRA)
-Hugging Face Datasets
-SQLite
-Google Colab
-YAML
