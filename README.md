# cs-exam-coach-qlora

A small QLoRA + SFT practice project for building a Traditional Chinese CS exam assistant.

## Project Goal

This project fine-tunes a small instruction model with a conversational dataset so that it answers CS exam concepts in a fixed format:

1. One-sentence definition
2. Core concept
3. Common misconception
4. Simple example

## Project Structure

```text
data/
  train.jsonl
  eval.jsonl

scripts/
  build_dataset.py
  train_qlora.py
  infer.py
