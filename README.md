
# Advanced LLM Fine-Tuning Framework (QLoRA / LoRA / Full Fine-Tune)

**Author:** Mohammad Kahab  
**Role:** Machine Learning Engineer  
**Location:** Malappuram, Kerala, India  
**Version:** 1.0 (March 2026)  
**Purpose:** Production-grade, single-file, fully customizable fine-tuning script with **auto-preprocessing**, seamless model switching, QLoRA/LoRA/full fine-tuning, high-end optimizations (Flash Attention 2, gradient checkpointing, 4-bit double quant, bf16, long-context support, bi-directional prompt formatting, etc.).

---

## Features (Everything Built-In)

- **One-click model switching** – Just change `--model_name_or_path`
- **Auto preprocessing** – Supports Alpaca, ShareGPT, ChatML, raw text, and custom formats
- **Three training modes**:
  - **QLoRA** (4-bit NF4 + double quant – lowest VRAM)
  - **LoRA** (16-bit / bf16 – half fine-tunable)
  - **Full fine-tuning** (no PEFT – maximum quality)
- **Bi-contextualizable prompts** – Automatically detects instruction/input/output or chat format and builds bidirectional-style context (perfect for long-context models)
- High-end techniques included:
  - Flash Attention 2 (if installed)
  - Gradient checkpointing + gradient accumulation
  - 4-bit double quantization + NF4
  - Fully mergeable adapters after training
  - Wandb / TensorBoard / Comet logging
  - Push to Hub (private/public)
- Zero extra files needed – everything in **one script**

---

## Quick Start

### 1. Create project folder
```bash
mkdir advanced-finetune-mohammad && cd advanced-finetune-mohammad
```

### 2. Save the two files below
- `fine_tune.py` (copy the big code block)
- `README.md` (this file)

### 3. Install dependencies
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install "transformers>=4.48.0" "datasets>=3.0.0" "peft>=0.13.0" "trl>=0.11.0" accelerate bitsandbytes scipy wandb flash-attn --no-build-isolation
```

### 4. Run example (Llama-3.1-8B QLoRA on Dolly)
```bash
python fine_tune.py \
  --model_name_or_path "meta-llama/Llama-3.1-8B-Instruct" \
  --dataset_name "databricks/databricks-dolly-15k" \
  --output_dir "./results-llama3.1-qlora" \
  --use_qlora \
  --lora_r 64 \
  --lora_alpha 16 \
  --max_seq_length 8192 \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-4 \
  --push_to_hub \
  --hub_model_id "mohammadkahab/llama3.1-8b-dolly-qlora" \
  --report_to wandb
```

---

## Full Command-Line Options (All Built-In)

```bash
python fine_tune.py --help
```

**Key flags:**
- `--use_qlora` → QLoRA (recommended)
- `--use_lora` → Regular LoRA (half-precision)
- `--full_finetune` → Full fine-tuning (no adapters)
- `--dataset_format` → `alpaca` | `sharegpt` | `chatml` | `text` (auto-detected if not given)
- `--max_seq_length` → Supports 32k+ context (Llama-3.1, Mistral, Qwen2, etc.)
- `--merge_after_training` → Automatically merges adapters and saves full model
- `--flash_attn` → Enables Flash Attention 2

---

## Supported Models (Tested & Ready)

Any HF model:
- Llama-3.1 / 3 / 2
- Mistral / Mixtral
- Qwen2 / Qwen2.5
- Gemma-2
- Phi-3 / 4
- DeepSeek, Command-R, etc.

Just pass the HF path – script handles quantization automatically.

---

## Folder Structure After Training

```
results-your-model/
├── adapter_model/          # LoRA/QLoRA weights (tiny)
├── merged_model/           # (if --merge_after_training)
├── trainer_state.json
├── training_args.bin
└── README.md (auto-generated)
```

---

## Why This Script is High-End

- Built with **TRL + PEFT + BitsAndBytes + Accelerate** (2026 best practices)
- Bi-contextual formatting → treats instruction + input as full bidirectional context for better long-context learning
- Fully parameter-efficient + full fine-tune in same codebase
- Zero manual tokenization – auto chat template + packing
- Ready for 1×A100, 1×H100, or multi-GPU (just set `device_map="auto"`)

---

