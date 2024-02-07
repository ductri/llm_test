import numpy as np
import pandas as pd

import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from our import constants
from direct_preference_optimization.utils import disable_dropout


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


if __name__ == "__main__":

    device = 'cuda'
    trained_model_path = f'{constants.ROOT}/.cache/nguyetr9/sentiment_controlled_env_2024-02-03_21-05-55_792456/LATEST/policy.pt'
    init_model_path = '/nfs/stak/users/nguyetr9/hpc-share/llm_test/models/gpt2-imdb/checkpoint-700/'
    # ref_model = load_ref_model(init_model_path)

    tokenizer = AutoTokenizer.from_pretrained(init_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
        'num_return_sequences': 1,
    }
    # model = load_model_dpo_way(init_model_path, trained_model_path)
    model = load_ref_model(init_model_path)
    model = model.to(device)
    text = ['Watching this']
    data = tokenizer(text)
    tensor_input = torch.tensor(data['input_ids']).to(device)
    output = model.generate(tensor_input, **generation_kwargs)
    output_text = tokenizer.decode(output[0])
    print(output_text)

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
    sent_kwargs = {"top_k": None, "function_to_apply": "softmax", "batch_size": 16}
    outputs = sentiment_pipeline([output_text], **sent_kwargs)
    print(outputs)
    print()

