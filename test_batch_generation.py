import itertools
from typing import Any
import numpy as np
import pandas as pd

import torch
import transformers
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler
from datasets import load_dataset

from our import constants
from direct_preference_optimization.utils import disable_dropout
from our.sentiment_gt import SentimentGT
from our.evaluate_alignment import load_ref_model, my_generation



if __name__ == "__main__":
    sentiment_gt = SentimentGT()

    init_model_path = '/nfs/stak/users/nguyetr9/hpc-share/llm_test/models/gpt2-imdb/checkpoint-700/'
    model = load_ref_model(init_model_path)

    tokenizer = AutoTokenizer.from_pretrained(init_model_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'left'

    batch_resp_text = []
    dataset = ['I love',  'I hate this']
    all_scores = []
    all_resps = []
    device = 'cuda'
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": False,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
        'num_return_sequences': 1,
    }
    for sample in tqdm(dataset, total=len(dataset)):
        input_ids = torch.tensor(tokenizer(sample)['input_ids']).unsqueeze(dim=0).to(device)
        attention_mask = torch.ones(input_ids.shape, dtype=torch.int32).to(device)
        resp = model.generate(input_ids, attention_mask=attention_mask, **generation_kwargs)[0]
        resp_text = tokenizer.decode(resp, skip_special_tokens=True)
        all_resps.append(resp_text)
        all_scores.append(sentiment_gt.get_score([resp_text])[0])
    all_scores = np.array(all_scores)
    print(all_resps)
    print()

    print('batching ...')
    resps_text = my_generation(model, tokenizer, dataset)
    batch_scores = sentiment_gt.get_score(resps_text)
    print(resps_text)
    __import__('pdb').set_trace()
    print()

