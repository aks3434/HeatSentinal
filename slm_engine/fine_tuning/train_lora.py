"""
HeatSentinel — SLM Fine-Tuning Script with PEFT / LoRA
Fine-tunes a Small Language Model (Phi-3 Mini or Qwen2-1.5B) on IMD Heat Action Plans.
Run this on GPU (local RTX or Google Colab T4).
"""

import os
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
CORPUS_PATH = DATA_DIR / "slm_finetune_corpus.jsonl"
OUTPUT_DIR = DATA_DIR / "slm_lora_adapter"

BASE_MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


def run_lora_finetuning():
    """Runs PEFT LoRA instruction fine-tuning on the generated corpus."""
    if not CORPUS_PATH.exists():
        print(f"❌ Error: Corpus {CORPUS_PATH} not found. Run dataset_generator.py first!")
        return

    print(f"🚀 Preparing LoRA Fine-Tuning for: {BASE_MODEL_ID}")
    print(f"📁 Training Corpus: {CORPUS_PATH}")
    print(f"💾 Output Adapter Directory: {OUTPUT_DIR}")

    try:
        import torch
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"⚡ Compute Device: [{device}]")

        # 1. Load Dataset
        dataset = load_dataset("json", data_files=str(CORPUS_PATH), split="train")

        def formatting_prompts_func(example):
            output_texts = []
            for i in range(len(example["instruction"])):
                text = (
                    f"<|user|>\n{example['instruction'][i]} {example['input'][i]}<|end|>\n"
                    f"<|assistant|>\n{example['output'][i]}<|end|>"
                )
                output_texts.append(text)
            return output_texts

        # 2. Tokenizer & Model
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )

        # 3. LoRA Configuration
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

        # 4. Training Arguments
        training_args = TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            num_train_epochs=3,
            logging_steps=5,
            fp16=torch.cuda.is_available(),
            save_strategy="epoch"
        )

        trainer = SFTTrainer(
            model=model,
            train_dataset=dataset,
            peft_config=peft_config,
            formatting_func=formatting_prompts_func,
            args=training_args,
        )

        print("🔥 Starting SFTTrainer training loop...")
        trainer.train()
        trainer.save_model(str(OUTPUT_DIR))
        print(f"✅ Successfully saved LoRA adapter weights to: {OUTPUT_DIR}")

    except ImportError as e:
        print(f"⚠️ PyTorch / HuggingFace dependencies missing: {e}")
        print("To install: pip install transformers peft trl datasets accelerate")
    except Exception as e:
        print(f"❌ Training error: {e}")


if __name__ == "__main__":
    run_lora_finetuning()
