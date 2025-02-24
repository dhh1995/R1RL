from functools import partial
from typing import Any, Callable, Dict, List

from datasets import Dataset, load_dataset
from loguru import logger
from termcolor import colored

from r1rl.prompt import ANSWER_TAG, REASONING_TAG, display_prompt
from r1rl.utils import extract_answer

from .base import BaseDataset

SYSTEM_PROMPT = f"""
For the given question, think about the reasoning process in the mind, and respond in the following format:
<{REASONING_TAG}>
reasoning process here
</{REASONING_TAG}>
<{ANSWER_TAG}>
answer here, contains only a single integer.
</{ANSWER_TAG}>
""".strip()

# SYSTEM_PROMPT = """
# A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
# The assistant first thinks about the reasoning process in the mind and then provides the user
# with the answer. The reasoning process and answer are enclosed within <think> </think> and
# <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
# <answer> answer here </answer>. User: {prompt}. Assistant:
# """.strip()


class GSM8kDataset(BaseDataset):
    def __init__(
        self,
        env_reward_scale: float = 1.0,
        eval_count: int | None = None,
        remove_comma: bool = True,
        is_chat: bool = True,
        shuffle_seed: int | None = None,
        add_reasoning_prefix: bool = False,
    ):
        self.dataset = load_dataset("openai/gsm8k", "main")
        self.env_reward_scale = env_reward_scale
        self.remove_comma = remove_comma
        self.is_chat = is_chat
        self.shuffle_seed = shuffle_seed
        self.add_reasoning_prefix = add_reasoning_prefix

        # Load and prep dataset
        self.raw_train_dataset = self.dataset["train"]
        self._train_dataset = self.process_dataset(
            self.raw_train_dataset,
        )
        self.raw_eval_dataset = self.dataset["test"]
        self._eval_dataset = self.process_dataset(
            self.raw_eval_dataset,
            count=eval_count,
        )
        self._reward_funcs = [self.correctness_reward_func]

    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @property
    def eval_dataset(self) -> Dataset:
        return self._eval_dataset

    @property
    def reward_funcs(self) -> List[Callable]:
        return self._reward_funcs

    def extract_hash_answer(self, text: str) -> str | None:
        """extract the answer from the text for GSM8K dataset"""
        if "####" not in text:
            return None
        ans = text.split("####")[1].strip()
        if self.remove_comma:
            ans = ans.replace(",", "")  # remove commas to unify the format
        return ans

    def build_prompt(self, question: str) -> List[Dict[str, str]] | str:
        if self.is_chat:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
            if self.add_reasoning_prefix:
                messages.append({"role": "assistant", "content": f"<{REASONING_TAG}>"})
        else:
            prompt = SYSTEM_PROMPT + "\nQuestion: " + question
            if self.add_reasoning_prefix:
                prompt = prompt + f"\n<{REASONING_TAG}>"
            return prompt

    def build_task_dict(self, task: Dict[str, str]) -> Dict[str, Any]:
        return {
            "prompt": self.build_prompt(task["question"]),
            "answer": self.extract_hash_answer(task["answer"]),
            "question": task["question"],
        }

    def process_dataset(
        self,
        dataset: Dataset,
        count: int | None = None,
    ) -> Dataset:
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)
        if count is not None:
            dataset = dataset.select(range(min(count, len(dataset))))
        return dataset.map(self.build_task_dict)

    # Reward functions
    def correctness_reward_func(
        self,
        completions: List[List[Dict[str, str]] | List[str]],
        *,
        prompts: List[List[Dict[str, str]] | List[str]],
        answer: List[str],
        question: List[str],
        **kwargs,
    ) -> list[float]:
        def get_completion_content(completion: List[Dict[str, str]] | str) -> str:
            if isinstance(completion, str):
                return completion
            else:
                return completion[0]["content"]

        responses = [get_completion_content(completion) for completion in completions]
        extracted_responses = [extract_answer(r) for r in responses]
        infos = {
            "Prompt": display_prompt(prompts[0]),
            "Question": question[0],
            "Response": colored(responses[0], "grey"),
            "Ground Truth": answer[0],
            "Extracted": "\n".join(
                [f"{i + 1}. {r}" for i, r in enumerate(extracted_responses)]
            ),
        }
        logger.info(
            "\n".join([f"{colored(k, 'green')}:\n{v}" for k, v in infos.items()])
        )
        return [
            self.env_reward_scale if r == a else 0.0
            for r, a in zip(extracted_responses, answer)
        ]
