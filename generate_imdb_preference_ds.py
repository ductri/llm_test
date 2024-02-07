import itertools
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import torch
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from trl.core import LengthSampler

from our import constants
from our.evaluate_alignment import my_generation
from our.sentiment_gt import SentimentGT


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
    tokenizer = AutoTokenizer.from_pretrained(f'{constants.ROOT}/models/gpt2-imdb/checkpoint-700/')
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = build_dataset(tokenizer)

    sentiment_gt = SentimentGT()

    batch_size = 256

    # sentiment_model = AutoModelForSequenceClassification.from_pretrained(
    #         pretrained_model_name_or_path="siebert/sentiment-roberta-large-english",
    #         cache_dir=f'{constants.ROOT}/.cache/')
    # sentiment_tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english",
    #         cache_dir=f'{constants.ROOT}/.cache/')
    # sentiment_pipeline = pipeline("sentiment-analysis",
    #         model=sentiment_model,
    #         tokenizer=tokenizer,
    #         device=device)
    # sent_kwargs = {"top_k": None, "function_to_apply": "softmax", "batch_size": batch_size}

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True, collate_fn=collator)
    num_return_sequences = 4
    max_new_tokens = 16
    all_data = []
    for batch in tqdm(train_loader):
        text_queries = batch['query']
        max_input_length = 8
        text_resps = my_generation(imdb_model, tokenizer, text_queries,
                max_input_length=max_input_length,
                do_sample=True, num_return_sequences=num_return_sequences,
                include_input=False)
        text_queries = itertools.chain(*[[item]*num_return_sequences for item in text_queries])
        full_sentences = [p+r for p, r in zip(text_queries, text_resps)]
        assert len(full_sentences) == batch_size*num_return_sequences
        sentiment_scores = sentiment_gt.get_score(full_sentences)

        for i in range(len(batch['query'])):
            prompt = batch['query'][i]
            list_pairs = itertools.combinations(range(i*num_return_sequences, (i+1)*num_return_sequences), 2)
            list_pairs = list(list_pairs)
            for pair in list_pairs:
                res1_text = text_resps[pair[0]]
                res2_text = text_resps[pair[1]]
                sentiment_score1 = sentiment_scores[pair[0]]
                sentiment_score2 = sentiment_scores[pair[1]]
                if sentiment_score1 > sentiment_score2:
                    all_data.append((prompt, res1_text, sentiment_score1, res2_text, sentiment_score2))
                else:
                    all_data.append((prompt, res2_text, sentiment_score2, res1_text, sentiment_score1))

    df = pd.DataFrame(all_data, columns=['prompt',  'good_response',  'good_response_score',  'bad_response',  'bad_response_score'])
    Path('./data').mkdir(exist_ok=True)
    path_to_file = 'data/sentiment_imdb_preference_dataset.csv'
    df.to_csv(path_to_file, index=None)
    print(f'Saved perference dataset to `{path_to_file}`')


