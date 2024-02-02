import torch
from tqdm import tqdm
import pandas as pd

tqdm.pandas()

from transformers import pipeline, AutoTokenizer
from datasets import load_dataset

from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb

import utils


def build_dataset(config, dataset_name="imdb", input_min_text_length=2, input_max_text_length=8):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        dataset_name (`str`):
            The name of the dataset to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    # load imdb with datasets
    ds = load_dataset(dataset_name, split="train")
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")
    return ds


def collator(data):
    return dict((key, [d[key] for d in data]) for key in data[0])


if __name__ == "__main__":
    batch_size=512
    config = PPOConfig(
        model_name="lvwerra/gpt2-imdb",
        learning_rate=5e-7,
        log_with="wandb",
        batch_size=batch_size,
    )

    sent_kwargs = {"return_all_scores": True, "function_to_apply": "none", "batch_size": batch_size}
    wandb.init()

    dataset = build_dataset(config)

    model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
    print(f'#params: {utils.count_parameters(model)}')
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    tokenizer.pad_token = tokenizer.eos_token
    ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=dataset, data_collator=collator)
    device = ppo_trainer.accelerator.device
    if ppo_trainer.accelerator.num_processes == 1:
        device = 0 if torch.cuda.is_available() else "cpu"  # to avoid a `pipeline` bug
    sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)
    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}
    output_min_length = 4
    output_max_length = 16
    output_length_sampler = LengthSampler(output_min_length, output_max_length)


    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
    }


    for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader), total=len(ppo_trainer.dataloader)):
        query_tensors = batch["input_ids"]
        print('got a new batch')

        # #### Get response from gpt2
        # response_tensors = []
        # for i, query in enumerate(query_tensors):
        #     gen_len = output_length_sampler()
        #     generation_kwargs["max_new_tokens"] = gen_len
        #     response = ppo_trainer.generate(query, **generation_kwargs)
        #     response_tensors.append(response.squeeze()[-gen_len:])
        # batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

        # Get response from gpt2
        response_tensors, ref_response_tensors = ppo_trainer.generate(
            query_tensors, return_prompt=False, generate_ref_response=True, **generation_kwargs
        )
        batch["response"] = tokenizer.batch_decode(response_tensors)
        batch["ref_response"] = tokenizer.batch_decode(ref_response_tensors)
        print('got all responses')

        # Compute sentiment score
        texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
        ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
        ref_pipe_outputs = sentiment_pipe(ref_texts, **sent_kwargs)
        ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
        batch["ref_rewards"] = ref_rewards
        print('got reward')



        #### Compute sentiment score
        # texts = [q + r for q, r in zip(batch["query"], batch["response"])]
        # pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
        # rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]

        print('running ppo')
        #### Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])

    #### Save model
    ppo_trainer.save_model(f'{constants.ROOT}/models/gpt2-positive/')

