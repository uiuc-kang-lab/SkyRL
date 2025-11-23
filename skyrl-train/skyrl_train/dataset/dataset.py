import datasets
from loguru import logger
import os
from copy import deepcopy
from typing import List
from transformers import PreTrainedTokenizerBase


class PromptDataset:
    def __init__(
        self,
        datasets: str | List[str],
        tokenizer: PreTrainedTokenizerBase,
        max_prompt_length: int,
        num_workers: int = 8,
        prompt_key: str = "prompt",
        env_class_key: str = "env_class",
        noise_level: float = 0.0,
        **kwargs,
    ):
        self.tokenizer = tokenizer
        self.max_prompt_length = max_prompt_length
        self.prompt_key = prompt_key
        self.env_class_key = env_class_key
        self.num_workers = num_workers
        self.noise_level = noise_level

        self.datasets = datasets
        if isinstance(self.datasets, str):
            self.datasets = [self.datasets]

        self._read_files_and_tokenize()

    def _introduce_noise(self):
        logger.info(f"Noise level: {self.noise_level}")
        if self.noise_level > 0:
            # check the first example to find out the noise model
            logger.info(f"Introducing noise to prompts with noise level {self.noise_level}")
            noise_model = self.dataframe[0]["noise_spec"]["method"]
            # if noise_model == "randomly_generate":
            #     logger.info(f"Using uniform-random noise model")
            #     # need to find min and max of the ground truth
            #     ground_truths = [int(item["reward_spec"]["ground_truth"]) for item in self.dataframe]
            #     min_gt = min(ground_truths)
            #     max_gt = max(ground_truths)
            # loop over the items in the dataset
            def add_noise(example):
                import random
                if float(example["noise_spec"]["param"]) < self.noise_level:
                    if noise_model == "randomly_generate":
                        ground_truth = int(example["reward_spec"]["ground_truth"])
                        min_noise = max(0, ground_truth - 3)
                        max_noise = ground_truth + 3
                        noisy_answer = random.randint(min_noise, max_noise)
                        example["reward_spec"]["ground_truth"] = str(noisy_answer)
                    elif noise_model == "randomly_remove":
                        example["reward_spec"]["ground_truth"] = deepcopy(example["reward_spec"]["insufficient_tests"])
                    elif noise_model == "randomly_replace":
                        example["reward_spec"]["ground_truth"] = deepcopy(example["reward_spec"]["wrong_answer"])
                    else:
                        raise ValueError(f"Unknown noise model: {noise_model}")
                return example
            # print some examples to verify the noise are added
            for i in range(20):
                if len(self.dataframe[0]["reward_spec"]["ground_truth"]) > 100:
                    logger.info(f"Before noise model: total {len(self.dataframe[i]['reward_spec']['ground_truth'])} tests")
                else:
                    logger.info(f"Before noise model: {self.dataframe[i]['reward_spec']}")
            self.dataframe = self.dataframe.map(add_noise, num_proc=self.num_workers, desc="Adding noise to prompts")

            for i in range(20):
                if len(self.dataframe[0]["reward_spec"]["ground_truth"]) > 100:
                    logger.info(f"After noise model: total {len(self.dataframe[i]['reward_spec']['ground_truth'])} tests")
                else:
                    logger.info(f"After noise model: {self.dataframe[i]['reward_spec']}")
        else:
            logger.info("No noise added to reward_spec")

    def _read_files_and_tokenize(self):
        loaded_datasets = []
        for source in self.datasets:
            ext = os.path.splitext(source)[-1].lower()
            if ext == ".parquet":
                ds = datasets.load_dataset("parquet", data_files=source, keep_in_memory=True)["train"]
            elif ext in [".json", ".jsonl"]:
                ds = datasets.load_dataset("json", data_files=source, keep_in_memory=True)["train"]
            else:
                # Treat as HF dataset spec: "name" or "name:split"
                dataset_name, has_split, split = source.partition(":")
                try:
                    ds_dict = datasets.load_dataset(path=dataset_name, keep_in_memory=True)
                except ValueError:
                    raise ValueError(f"Dataset `{dataset_name}` not found on Hugging Face.")
                split = split if has_split else "train"
                if split not in ds_dict:
                    raise ValueError(
                        f"Split `{split}` not found in dataset `{dataset_name}`. Configured split was `{split}` and default is `train`"
                    )
                ds = ds_dict[split]
            loaded_datasets.append(ds)

        self.dataframe: datasets.Dataset = datasets.concatenate_datasets(loaded_datasets)

        logger.info(f"dataset len: {len(self.dataframe)}")
        
        # calculate the max length of the prompts
        lengths = self.dataframe.map(
            lambda doc: {"len": len(self.tokenizer.apply_chat_template(doc[self.prompt_key], add_generation_prompt=True))},
            num_proc=self.num_workers,
            desc="Calculating prompt lengths",
        )["len"]
        max_length = max(lengths)
        logger.info(f"Max prompt length in dataset: {max_length} tokens")
        logger.info(f"Total dataset size: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe.filter(
            lambda doc: len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))
            <= self.max_prompt_length,
            num_proc=self.num_workers,
            desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
        )

        logger.info(f"Filtered dataset size: {len(self.dataframe)}")

        # introduce noise if needed
        self._introduce_noise()

    def __getitem__(self, item):
        row_dict: dict = self.dataframe[item]
        messages = row_dict.pop(self.prompt_key)
        env_class = row_dict.pop(self.env_class_key, None)

        extra = {key: value for key, value in row_dict.items() if key != self.prompt_key and key != self.env_class_key}
        uid = str(item)

        return messages, env_class, extra, uid

    def collate_fn(self, item_list):
        all_inputs = []
        for prompt, env_class, env_extras, item_uids in item_list:
            all_inputs.append({"prompt": prompt, "env_class": env_class, "env_extras": env_extras, "uid": item_uids})
        return all_inputs

    def __len__(self):
        return len(self.dataframe)
