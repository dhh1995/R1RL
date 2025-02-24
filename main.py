import argparse
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import wandb
from loguru import logger
from unsloth import FastLanguageModel, PatchFastRL, is_bfloat16_supported

PatchFastRL("GRPO", FastLanguageModel)

from trl import GRPOConfig, GRPOTrainer  # need to import after unsloth patched

from r1rl.datasets import GSM8kDataset


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dumps", "-du", type=str, default="dumps", help="dumps directory"
    )
    parser.add_argument(
        "--project-name", "-pn", type=str, default="r1-rl", help="project name"
    )
    parser.add_argument(
        "--dataset-name",
        "-dn",
        type=str,
        choices=["gsm8k"],
        help="dataset name",
        default="gsm8k",
    )
    parser.add_argument("--exp-name", "-en", type=str, default=None)
    parser.add_argument("--model-name", "-m", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument(
        "--gpu-memory-utilization",
        "-gmu",
        type=float,
        default=0.6,
        help="gpu memory utilization",
    )
    parser.add_argument(
        "--lora-rank",
        "-lora",
        type=int,
        default=64,
        help="rank of LoRA, larger rank = smarter, but slower",
    )
    parser.add_argument(
        "--max-seq-length",
        "-msl",
        type=int,
        default=1024,
        help="max sequence length, can increase for longer reasoning traces",
    )
    parser.add_argument(
        "--env-reward-scale",
        "-ers",
        type=float,
        default=1.0,
        help="environment reward scale",
    )
    parser.add_argument(
        "--learning-rate",
        "-lr",
        type=float,
        default=1e-5,
        help="learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        "-wd",
        type=float,
        default=0.1,
        help="weight decay",
    )
    parser.add_argument(
        "--warmup-ratio",
        "-wr",
        type=float,
        default=0.1,
        help="warmup ratio",
    )
    parser.add_argument(
        "--per-device-train-batch-size",
        "-tbs",
        type=int,
        default=8,
        help="per device train batch size, try to increase if out of memory",
    )
    parser.add_argument(
        "--per-device-eval-batch-size",
        "-ebs",
        type=int,
        default=32,
        help="per device eval batch size",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "-gas",
        type=int,
        default=1,
        help="gradient accumulation steps. When use small batch size, try to increase for smoother training",
    )
    parser.add_argument(
        "--num-generations",
        "-ngen",
        type=int,
        default=8,
        help="number of generations, decrease if out of memory",
    )
    parser.add_argument(
        "--max-prompt-length",
        "-mplen",
        type=int,
        default=256,
        help="max prompt length",
    )
    parser.add_argument(
        "--max-completion-length",
        "-mclen",
        type=int,
        default=750,
        help="max completion length",
    )
    parser.add_argument(
        "--max-steps",
        "-mstep",
        type=int,
        default=500,
    )
    parser.add_argument(
        "--eval-on-start",
        action="store_true",
        help="eval on start",
    )
    parser.add_argument(
        "--eval-count",
        type=int,
        default=None,
        help="eval count",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="eval steps",
    )
    parser.add_argument(
        "--save-steps",
        type=int,
        default=250,
        help="save steps",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.1,
        help="max grad norm",
    )
    parser.add_argument(
        "--not-remove-comma",
        action="store_true",
        help="not remove comma for the answer of gsm8k",
    )
    parser.add_argument(
        "--is-chat",
        "-chat",
        action="store_true",
        help="use chat format for the prompt, used for instruction fine-tuned models",
    )
    parser.add_argument(
        "--add-reasoning-prefix",
        "-rp",
        action="store_true",
        help="add reasoning prefix for the prompt",
    )
    args = parser.parse_args()

    if args.exp_name is None:
        args.exp_name = (
            f"{args.dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    if not os.path.exists(args.dumps):
        os.makedirs(args.dumps)

    return args


def main():
    args = get_args()
    logger.add(f"logs/logs_{args.exp_name}.log", level="INFO")
    logger.info(f"cmd: {Path(sys.executable).stem} {' '.join(sys.argv)}")
    logger.info(f"args: {args}")

    try:
        # get git commit hash
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"])
            .decode("utf-8")
            .strip()
        )
        logger.info(f"git commit hash: {commit_hash}")
    except Exception as e:
        logger.warning(f"failed to get git commit hash: {e}")

    wandb.init(project=args.project_name, name=args.exp_name, sync_tensorboard=True)

    dataset = GSM8kDataset(
        env_reward_scale=args.env_reward_scale,
        eval_count=args.eval_count,
        remove_comma=not args.not_remove_comma,
        is_chat=args.is_chat,
        add_reasoning_prefix=args.add_reasoning_prefix,
        shuffle_seed=3407,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,  # False for LoRA 16bit
        fast_inference=True,  # Enable vLLM fast inference
        max_lora_rank=args.lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,  # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],  # Remove QKVO if out of memory
        lora_alpha=args.lora_rank,
        use_gradient_checkpointing="unsloth",  # Enable long context finetuning
        random_state=3407,
    )

    training_args = GRPOConfig(
        use_vllm=True,  # use vLLM for fast inference!
        learning_rate=args.learning_rate,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        optim="adamw_8bit",
        logging_steps=1,
        bf16=is_bfloat16_supported(),
        fp16=not is_bfloat16_supported(),
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        max_steps=args.max_steps,
        eval_on_start=args.eval_on_start,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_grad_norm=args.max_grad_norm,
        report_to="wandb",  # Can use Weights & Biases
        output_dir=f"{args.dumps}/outputs_{args.exp_name}",
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=dataset.reward_funcs,
        args=training_args,
        train_dataset=dataset.train_dataset,
        eval_dataset=dataset.eval_dataset,
    )
    trainer.train()


if __name__ == "__main__":
    main()
