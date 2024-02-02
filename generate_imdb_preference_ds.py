import itertools

import pandas as pd
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, pipeline
from torch.utils.data import DataLoader
from trl.core import LengthSampler

import constants


def build_dataset(tokenizer):
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
    # load imdb with datasets
    ds = load_dataset('imdb', split="train", cache_dir=f'{constants.ROOT}/.cache/')
    ds = ds.select(range(100))
    ds = ds.rename_columns({"text": "review"})
    ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

    input_size = LengthSampler(2, 8)

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
    device = 'cuda'
    imdb_model = GPT2LMHeadModel.from_pretrained(f'{constants.ROOT}/models/gpt2-imdb/checkpoint-700/')
    imdb_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    tokenizer.pad_token = tokenizer.eos_token
    train_dataset = build_dataset(tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, collate_fn=collator)
    num_return_sequences = 4
    generation_kwargs = {
        "min_length": -1,
        "top_k": 0.0,
        "top_p": 1.0,
        "do_sample": True,
        "pad_token_id": tokenizer.eos_token_id,
        "max_new_tokens": 16,
        "num_return_sequences": num_return_sequences,
    }
    all_data = []
    sentiment_pipeline = pipeline("sentiment-analysis",
            model="siebert/sentiment-roberta-large-english",
            device=device,
            cache_dir=f'{constants.ROOT}/.cache/'
            )
    for batch in train_loader:
        query_tensors = batch['input_ids']
        max_length = 8
        padding_tensor = tokenizer.encode(tokenizer.pad_token)[0]*torch.ones(max_length, dtype=torch.int32)
        batch_attention_mask = []
        for i, query in enumerate(query_tensors):
            query_tensors[i] = torch.concat((padding_tensor, query))[-max_length:]
            attention_mask = torch.zeros(8, dtype=torch.int32)
            attention_mask[-query.shape[0]:] = 1
            batch_attention_mask.append(attention_mask)
        print()
        batch_attention_mask = torch.stack(batch_attention_mask).to(device)
        query_tensors_batch = torch.stack(query_tensors).to(device)
        outputs = imdb_model.generate(query_tensors_batch, attention_mask=batch_attention_mask, **generation_kwargs)
        __import__('pdb').set_trace()
        sentiment_scores = sentiment_pipeline(outputs)

        for i in range(batch_size):
            prompt = batch['query'][i]
            list_pairs = list(itertools.combinations(
                range(i*num_return_sequences, (i+1)*num_return_sequences), 2))
            for pair in list_pairs:
                res1_text = outputs[pair[0]]
                res2_text = outputs[pair[1]]
                sentiment_score1 = sentiment_scores[pair[0]]
                sentiment_score2 = sentiment_scores[pair[1]]
                if sentiment_score1 > sentiment_score2:
                    all_data.append((prompt, res1_text, sentiment_score1, res2_text, sentiment_score2))
                else:
                    all_data.append((prompt, res2_text, sentiment_score2, res1_text, sentiment_score1))

    df = pd.DataFrame(all_data, columns=['prompt',  'good_response',  'good_response_score',  'bad_response',  'bad_response_score'])
    df.to_csv('data/sentiment_imdb_reference_dataset.csv', index=None)


