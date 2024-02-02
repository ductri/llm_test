"""
* Main reference:  https://github.com/huggingface/trl/blob/main/examples/notebooks/gpt2-sentiment.ipynb
* Resource:
- https://huggingface.co/docs/trl/main/en/ppo_trainer
"""


from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from pprint import pprint
import torch
from tqdm import tqdm
import pandas as pd
tqdm.pandas()
from transformers import pipeline, AutoTokenizer
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
import wandb

import constants
import utils


def build_dataset():
    dataset = load_dataset("imdb", split="train", cache_dir=f'{constants.ROOT}/.cache/')
    dataset = dataset.filter(lambda x: len(x["text"]) > 200, batched=False)

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    input_min_text_length = 2
    input_max_text_length = 8
    input_size = LengthSampler(input_min_text_length, input_max_text_length)
    def tokenize(element):
        element['input_ids'] = tokenizer.encode(element["text"])[: input_size()]
        element['query'] = tokenizer.decode(element["input_ids"])
        return element
    tokenized_dataset = dataset.map(tokenize, batched=False)
    tokenized_dataset.set_format(type="torch")
    print(f'Training set size: {len(tokenized_dataset)}')
    return tokenized_dataset



def main():
    wandb.init(project=constants.PROJ_NAME)
    batch_size = 256

    reward_model = pipeline('text-classification', model='lvwerra/distilbert-imdb')
    sent_kwargs = {"return_all_scores": True, "function_to_apply": "softmax", "batch_size": batch_size}


    def collator(data):
        return dict((key, [d[key] for d in data]) for key in data[0])

    init_model = AutoModelForCausalLMWithValueHead.from_pretrained(f'{constants.ROOT}/models/gpt2-imdb/checkpoint-700/')
    ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(f'{constants.ROOT}/models/gpt2-imdb/checkpoint-700/')

    ppo_config = PPOConfig(log_with='wandb', model_name='my-imdb-finetuned-gpt2',
            learning_rate=5e-7, init_kl_coef=0.2, adap_kl_ctrl=True,
            batch_size = batch_size, ppo_epochs=1,
    )
    tokenized_dataset = build_dataset()
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    ppo_trainer = PPOTrainer(
            model = init_model,
            config = ppo_config,
            dataset = tokenized_dataset,
            tokenizer = tokenizer,
            data_collator = collator,
            )


    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
    }

    # output_min_length = 4
    # output_max_length = 16
    # output_length_sampler = LengthSampler(output_min_length, output_max_length)

    for epoch in tqdm(range(ppo_trainer.config.ppo_epochs), "epoch: "):
        for batch in tqdm(ppo_trainer.dataloader):
            query_tensors = batch["input_ids"]
            __import__('pdb').set_trace()

            # #### Get response from gpt2
            # response_tensors = []
            # for query in query_tensors:
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

            #### Compute reward score
            texts = [q + r for q, r in zip(batch["query"], batch["response"])]
            pipe_outputs = reward_model(texts, **sent_kwargs)
            rewards = [torch.tensor(output[1]["score"]) for output in pipe_outputs]
            ref_texts = [q + r for q, r in zip(batch["query"], batch["ref_response"])]
            ref_pipe_outputs = reward_model(ref_texts, **sent_kwargs)
            ref_rewards = [torch.tensor(output[1]["score"]) for output in ref_pipe_outputs]
            batch["ref_rewards"] = ref_rewards

            #### Run PPO step
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            ppo_trainer.log_stats(stats, batch, rewards, columns_to_log=["query", "response", "ref_response", "ref_rewards"])


    #### Save model
    init_model.save_pretrained(f'{constants.ROOT}/models/gpt2-positive/')
    tokenizer.save_pretrained(f'{constants.ROOT}/models/gpt2-positive/')



if __name__ == "__main__":
    main()

