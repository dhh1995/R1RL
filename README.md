# Reproducing R1-Zero-like Reforcement Learning on GSM8K Using Unsloth

A simplified reproduction of R1-Zero-like Reinforcement Learning on GSM8K, based on the [notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb#scrollTo=vzOuSVCL_GA9) from [Unsloth](https://github.com/unsloth-ai/unsloth).

![training curve](_assets/gsm8k_train1.png)

The training curve above is for Qwen2.5-3B loaded with 4-bit quantization and LoRA rank 64 on a single A10 GPU (24GB).
The 200 training steps takes about 2 hours, with total cost less than $2.

## Install

```bash
# create your own conda environment
pip install unsloth vllm

# install the package
git clone https://github.com/dhh1995/R1-RL.git
pip install -e .
```

### Troubleshooting

see the following links if you encounter any issues:

- [Unsloth](https://docs.unsloth.ai/get-started/installing-+-updating)
- [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html)

## Training on GSM8K
Simply Run with:

```bash
python main.py --eval-on-start --eval-count 200 --eval-steps 100 --per-device-train-batch-size 16 --add-reasoning-prefix
```

The arguments are as follows:
```python
args: Namespace(dumps='dumps', project_name='r1-rl', dataset_name='gsm8k', exp_name='gsm8k', model_name='Qwen/Qwen2.5-3B', gpu_memory_utilization=0.5, lora_rank=64, max_seq_length=1024, env_reward_scale=1.0, learning_rate=1e-05, weight_decay=0.1, warmup_ratio=0.1, per_device_train_batch_size=16, per_device_eval_batch_size=32, gradient_accumulation_steps=1, num_generations=8, max_prompt_length=256, max_completion_length=750, max_steps=500, eval_on_start=True, eval_count=200, eval_steps=100, save_steps=250, max_grad_norm=0.1, not_remove_comma=False, is_chat=False, add_reasoning_prefix=True)
```

### Dataset

The dataset is from [GSM8K](https://huggingface.co/datasets/gsm8k) and processed in [`gsm8k.py`](./src/r1rl/datasets/gsm8k.py).
To reduce the eval time, we use a random subset of 200 examples from test set.
For 200 steps (in the training curve above) with batch size 16 and 8 generations per example, the number of training examples used is 400 (=200 * 16 / 8).

We observed that about 1% of the answer contains `,` in the numerical answer. We remove the comma when comparing the answer.

### Reward Function
Following [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/), we remove the reward functions for the correct formatting. We use the correctness reward function only (simply comparing the extracted answer with the ground truth). We observed that the formatting reward function is not necessary for this training.

### Model

We use [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B) for training and loaded with 4-bit quantization.

We also tried [Qwen/Qwen2.5-3B-Instruct](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) as the starting model, but we do not observe significant completion length change while reward being improved.

### The Prefix for Completion

When not adding the `<reasoning>` tag as the prefix for the completion, the completion length is generally longer with lower reward then the one with the prefix.

### Cheery-pick Results

LLM emerges a self-verification mechanism to check the correctness of the answer. see [full log](_assets/logs_gsm8k.log#28861).

```xml
Question: After Betty gave Stuart 40% of her marble collection, the number of marbles in Stuart's collection increased to 80. If Betty had 60 marbles, how many marbles did Stuart have initially?

<reasoning> First, let's find out how many marbles Betty gave to Stuart. Since she gave 40% of her collection, that would be 60 * 40% = 24 marbles. Now, we know that after giving away 24 marbles, the number of marbles in Stuart's collection increased to 80. So, before receiving the marbles, Stuart had 80 - 24 = 56 marbles. Since Betty gave him 40% of her collection to reach a total of 80 marbles, Stuart must have had 56 marbles initially. To verify, 40% of 60 marbles from Betty would be 24 marbles, and 56 + 24 = 80, which matches the given information. Therefore, the initial number of marbles Stuart had was 56. </reasoning>
<answer>56</answer>
```

### Full dataset results

We also trained the same model on A100 (40GB) with 1000 steps and larger batch size (32) and evaluated the model on the full test dataset (1.3k tests) and the results are shown below. The model can achieve accuracy on GSM8K full test set with more than 70% accuracy.

![training curve](_assets/gsm8k_train2.png)

Command:
```bash
python main.py --eval-on-start --max-steps 1000 --eval-steps 100 --per-device-train-batch-size 32 --per-device-eval-batch-size 64 --add-reasoning-prefix
```

## Next Steps
We are trying to apply this training to more datasets and models.

## Acknowledgement
We would like to thank [DeepSeek-R1-Zero](https://arxiv.org/abs/2501.12948) for showing the promising result of using Reinforcement Learning, and [Unsloth](https://github.com/unsloth-ai/unsloth) for [their blog](https://unsloth.ai/blog/r1-reasoning) and [colab notebook](https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Qwen2.5_(3B)-GRPO.ipynb#scrollTo=vzOuSVCL_GA9) that demonstrates a easy way to reproduce the R1-Zero-like training.

Also, we would like to thank [Open-Reasoner-Zero](https://github.com/Open-Reasoner-Zero/Open-Reasoner-Zero/) for ablating the reward function, and other open-source projects (like [TinyZero](https://github.com/Jiayi-Pan/TinyZero) and [SimpleRL-Zero](https://github.com/hkust-nlp/simpleRL-reason)) for sharing their knowledge.
