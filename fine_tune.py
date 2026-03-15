
# ======================== fine_tune.py ========================
# Advanced LLM Fine-Tuning Script - QLoRA / LoRA / Full
# Author: Mohammad Kahab (Machine Learning Engineer)
# ========================================================

import argparse
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer
import os
from datetime import datetime

def parse_args():
    parser = argparse.ArgumentParser(description="Mohammad Kahab's Advanced Fine-Tuning Framework")
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-3.1-8B-Instruct", help="HF model path")
    parser.add_argument("--dataset_name", type=str, required=True, help="HF dataset or local path")
    parser.add_argument("--dataset_format", type=str, default="auto", choices=["auto", "alpaca", "sharegpt", "chatml", "text"])
    parser.add_argument("--output_dir", type=str, default=f"./results-{datetime.now().strftime('%Y%m%d-%H%M')}")
    parser.add_argument("--max_seq_length", type=int, default=8192)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--use_qlora", action="store_true")
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--full_finetune", action="store_true")
    parser.add_argument("--merge_after_training", action="store_true")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--report_to", type=str, default="wandb", choices=["wandb", "tensorboard", "none"])
    parser.add_argument("--flash_attn", action="store_true")
    return parser.parse_args()

def get_formatting_func(dataset_format, tokenizer):
    def alpaca_format(example):
        if "input" in example and example["input"].strip():
            text = f"""<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
{example['instruction']}\n{example['input']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        else:
            text = f"""<|im_start|>user
{example['instruction']}<|im_end|>
<|im_start|>assistant
{example['output']}<|im_end|>"""
        return {"text": text}

    def sharegpt_format(example):
        text = ""
        for msg in example["conversations"]:
            role = "user" if msg["from"] == "human" else "assistant"
            text += f"<|im_start|>{role}\n{msg['value']}<|im_end|>\n"
        return {"text": text}

    def chatml_format(example):
        return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}

    def text_format(example):
        return {"text": example["text"] if isinstance(example["text"], str) else str(example)}

    if dataset_format == "alpaca" or (dataset_format == "auto" and "instruction" in load_dataset("json", data_files="dummy.json", split="train").column_names):
        return alpaca_format
    elif dataset_format == "sharegpt":
        return sharegpt_format
    elif dataset_format == "chatml":
        return chatml_format
    else:
        return text_format

def main():
    args = parse_args()
    print(f"🚀 Starting fine-tuning by Mohammad Kahab - Model: {args.model_name_or_path}")

    # ====================== QUANTIZATION & MODEL LOAD ======================
    if args.use_qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        )
        model = prepare_model_for_kbit_training(model)
        print("✅ QLoRA mode activated (4-bit)")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2" if args.flash_attn else "eager",
        )
        print("✅ Full / LoRA mode (bf16)")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ====================== PEFT CONFIG ======================
    if args.use_qlora or args.use_lora:
        peft_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        print("✅ LoRA/QLoRA adapters applied")
    elif args.full_finetune:
        print("✅ Full fine-tuning mode (no PEFT)")

    # ====================== DATASET & FORMATTING ======================
    dataset = load_dataset(args.dataset_name)
    if "train" not in dataset:
        dataset = dataset["train"].train_test_split(test_size=0.05)
    train_dataset = dataset["train"]

    formatting_func = get_formatting_func(args.dataset_format, tokenizer)

    # ====================== TRAINING ARGS ======================
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        fp16=False,
        bf16=True,
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to=args.report_to if args.report_to != "none" else None,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        gradient_checkpointing=True,
        packing=True,
        max_seq_length=args.max_seq_length,
    )

    # ====================== TRAINER ======================
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        formatting_func=formatting_func,
        args=training_args,
        max_seq_length=args.max_seq_length,
    )

    print("🔥 Starting training...")
    trainer.train()

    # ====================== SAVE & MERGE ======================
    trainer.save_model(args.output_dir)
    if (args.use_qlora or args.use_lora) and args.merge_after_training:
        print("🔄 Merging adapters...")
        model = model.merge_and_unload()
        model.save_pretrained(os.path.join(args.output_dir, "merged_model"))
        tokenizer.save_pretrained(os.path.join(args.output_dir, "merged_model"))
        print("✅ Model merged and saved!")

    print(f" Training completed! Results in: {args.output_dir}")
    

if __name__ == "__main__":
    main()
