from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import wandb
import evaluate
from evaluate import load

import utils
import constants


if __name__ == "__main__":
    wandb.init(project=constants.PROJ_NAME)
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")

    test_dataset = load_dataset("imdb", split="test", cache_dir=f'{constants.ROOT}/.cache/')
    test_dataset = test_dataset.select(range(5000))
    test_dataset = test_dataset.map(lambda x: {'text': x['text'][:1000]})

    perplexity = load("perplexity", module_type="metric")

    results = perplexity.compute(predictions=test_dataset['text'], model_id='gpt2-large', batch_size=128)
    ppl = results['mean_perplexity']
    print(f'PPL of GPT2: {ppl}')
    wandb.summary['ppl_0'] = ppl

    for i in range(100, 800, 100):
        model_path = f'{constants.ROOT}/models/gpt2-imdb/checkpoint-{i}'
        tokenizer.save_pretrained(model_path)

        results = perplexity.compute(predictions=test_dataset['text'], model_id=model_path, batch_size=128)
        ppl = results['mean_perplexity']
        print(f'PPL of GPT2 after finetuning: {ppl}')
        wandb.summary[f'ppl_{i}'] = ppl

    print()
    wandb.finish()
if a
