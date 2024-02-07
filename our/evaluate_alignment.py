import itertools
from typing import Any
import numpy as np
import pandas as pd

import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import load_dataset

from our import constants
from direct_preference_optimization.utils import disable_dropout
from our.sentiment_gt import SentimentGT


def evaluate_kl(ref_model, model, dataset):
    return 0.


def pad_batch_inds(list_inds_tensors, max_length, pad_token_id):
    padding_tensor = pad_token_id*torch.ones(max_length, dtype=torch.int32)
    query_tensors = torch.concat((padding_tensor, list_inds_tensors))[-max_length:]
    return query_tensors


def evaluate_sentiment(model, tokenizer, dataset, sentiment_pipeline):
    device = 'cuda'
    model = model.to(device)
    batch_size = 512
    sentiment_gt = SentimentGT()

    all_scores = []
    tokenizer.pad_token = tokenizer.eos_token
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset)//batch_size+1):
        batch = dataset[i: i+batch_size]
        text_resps = my_generation(model, tokenizer, batch['query'],
                max_input_length=8, num_return_sequences=1, do_sample=False,
                include_input=True)
        all_scores.extend(sentiment_gt.get_score(text_resps))
    all_scores = np.array(all_scores)
    print(f'Num examined samples: {len(all_scores)}')
    return all_scores.mean()


def evaluate_kl(model, ref_model, tokenizer, dataset):
    device = 'cuda'
    model = model.to(device)
    batch_size = 512
    sentiment_gt = SentimentGT()

    all_scores = []
    tokenizer.pad_token = tokenizer.eos_token
    for i in tqdm(range(0, len(dataset), batch_size), total=len(dataset)//batch_size+1):
        batch = dataset[i: i+batch_size]
        text_resps = my_generation(model, tokenizer, batch['query'],
                max_input_length=8, num_return_sequences=1, do_sample=False,
                include_input=True)
        all_scores.extend(sentiment_gt.get_score(text_resps))
    all_scores = np.array(all_scores)
    print(f'Num examined samples: {len(all_scores)}')
    return all_scores.mean()


def build_dataset(tokenizer):
    ds = load_dataset('imdb', split="train", cache_dir=f'{constants.ROOT}/.cache/')
    # ds = ds.select(range(1000))
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)
    input_size = LengthSampler(2, 8)
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    # def tokenize(text):
    #     return tokenizer(text, max_length=8, padding='max_length')

    # input_ids = [sample['input_ids'] for sample in ds]
    print(f'Dataset size after filtering: {len(ds)}')
    return ds


def load_model_dpo_way(init_model_path, trained_model_path):
    policy = transformers.AutoModelForCausalLM.from_pretrained(init_model_path)
    disable_dropout(policy)
    policy.eval()
    state_dict = torch.load(trained_model_path)
    policy.load_state_dict(state_dict['state'])
    return policy


def load_ref_model(init_model_path):
    policy = transformers.AutoModelForCausalLM.from_pretrained(init_model_path)
    disable_dropout(policy)
    policy.eval()
    device = 'cuda'
    policy = policy.to (device)
    return policy


def load_ppo_model():
    model = AutoModelForCausalLMWithValueHead.from_pretrained(f'{constants.ROOT}/models/ppo_gpt2_positive/')
    return model


def sample_pos_generation(prompts, models, model_names, sentiment_pipeline, tokenizer):
    device = 'cuda'
    for model in models:
        model.to(device)

    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
        'num_return_sequences': 1,
    }
    table = []
    sentiment_gt = SentimentGT()
    for prompt_text in tqdm(prompts):
        for _ in range(3):
            row = [prompt_text]
            prompt = tokenizer.encode(prompt_text)
            prompt = torch.tensor([prompt]).to(device)
            for model in models:
                output = model.generate(prompt, **generation_kwargs)
                output_text = tokenizer.decode(output[0])
                row.append(output_text)

                score = sentiment_gt.get_score([output_text])[0]
                row.append(score)
            table.append(row)
    columns =  ['prompt',] + list(itertools.chain(*[(name, name+'_score') for name in model_names]))
    test_table = wandb.Table(data=table, columns=columns)
    wandb.log({"sentiment_sample": test_table})


def my_generation(model, tokenizer, texts, max_input_length=8, num_return_sequences=1,
        do_sample=False, include_input=True):
    device = 'cuda'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'
    generation_kwargs = {
        "min_length": -1,
        "top_k": 50,
        "top_p": 1.0,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
        'num_return_sequences': num_return_sequences,
    }
    query_tensors = tokenizer(texts, max_length=max_input_length, padding='max_length',
            truncation=True)
    query_tensors['input_ids'] = torch.tensor(query_tensors['input_ids']).to(device)
    query_tensors['attention_mask'] = torch.tensor(query_tensors['attention_mask']).to(device)
    resp_tensor = model.generate(**query_tensors, **generation_kwargs)
    if not include_input:
        resp_tensor = resp_tensor[:, max_input_length:]
    resps_text = tokenizer.batch_decode(resp_tensor, skip_special_tokens=True)
    return resps_text



def main():
    run = wandb.init(project=constants.PROJ_NAME)
    run.tags = ['evaluation', 'dpo']
    device = 'cuda'

    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="siebert/sentiment-roberta-large-english",
            cache_dir=f'{constants.ROOT}/.cache/')
    sentiment_tokenizer = AutoTokenizer.from_pretrained(
            "siebert/sentiment-roberta-large-english",
            cache_dir=f'{constants.ROOT}/.cache/')
    sentiment_pipeline = pipeline("sentiment-analysis",
            model=sentiment_model,
            tokenizer=sentiment_tokenizer,
            device=device)

    trained_model_path = f'{constants.ROOT}/.cache/nguyetr9/sentiment_controlled_env_2024-02-06_16-44-06_839632/LATEST/policy.pt'
    init_model_path = '/nfs/stak/users/nguyetr9/hpc-share/llm_test/models/gpt2-imdb/checkpoint-700/'
    ref_model = load_ref_model(init_model_path)



    dpo_model = load_model_dpo_way(init_model_path, trained_model_path)
    dpo_model = dpo_model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(init_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    ppo_model = load_ppo_model()

    train_dataset = build_dataset(tokenizer)
    # ds = load_dataset('imdb', split="train", cache_dir=f'{constants.ROOT}/.cache/')
    train_kl, test_kl, train_sentiment_score, test_sentiment_score = 0,0,0,0

    print('--------------------------')
    print('Sampling')
    print('--------------------------')
    prompts = ['I watch',  'I think it was', 'First one was']
    sample_pos_generation(prompts, [ref_model, ppo_model, dpo_model],
            ['ref', 'ppo', 'dpo'], sentiment_pipeline, tokenizer)

    print('--------------------------')
    print('PPO MODEL')
    print('--------------------------')
    train_sentiment_score = evaluate_sentiment(ppo_model, tokenizer, train_dataset, sentiment_pipeline)
    summary = {'ppo_train_kl': train_kl, 'ppo_test_kl': test_kl,
            'ppo_train_sentiment_score': train_sentiment_score,
            'ppo_test_sentiment_score': test_sentiment_score,
            }
    print(f'train_kl={train_kl}, test_kl={test_kl}, '\
            f'train_sent_score={train_sentiment_score}, test_sent_score={test_sentiment_score}')
    for k, v in summary.items():
        wandb.summary[k] = v


    print('--------------------------')
    print('REF MODEL')
    print('--------------------------')
    train_sentiment_score = evaluate_sentiment(ref_model, tokenizer, train_dataset, sentiment_pipeline)
    summary = {'ref_train_kl': train_kl, 'ref_test_kl': test_kl,
            'ref_train_sentiment_score': train_sentiment_score,
            'ref_test_sentiment_score': test_sentiment_score,
            }
    print(f'train_kl={train_kl}, test_kl={test_kl}, '\
            f'train_sent_score={train_sentiment_score}, test_sent_score={test_sentiment_score}')
    for k, v in summary.items():
        wandb.summary[k] = v

    # test_dataset = ...

    # train_kl = evaluate_kl(ref_model, model, train_dataset)
    # test_kl = evaluate_kl(ref_model, model, test_dataset)
    #
    print('--------------------------')
    print('DPO MODEL')
    print('--------------------------')
    train_sentiment_score = evaluate_sentiment(dpo_model, tokenizer, train_dataset, sentiment_pipeline)
    # test_sentiment_score = evaluate_sentiment(model, test_dataset)

    summary = {'dpo_train_kl': train_kl, 'dpo_test_kl': test_kl,
            'dpo_train_sentiment_score': train_sentiment_score,
            'dpo_test_sentiment_score': test_sentiment_score,
            }
    for k, v in summary.items():
        wandb.summary[k] = v
    print()
    print(f'train_kl={train_kl}, test_kl={test_kl}, '\
            f'train_sent_score={train_sentiment_score}, test_sent_score={test_sentiment_score}')

