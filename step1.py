from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer
from transformers import Trainer, TrainingArguments, AutoModelWithLMHead
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
import wandb

import utils
import constants



def main():
    wandb.init(project='llm_test')

    train_dataset = load_dataset("imdb", split="train", cache_dir=f'{constants.ROOT}/.cache/')
    # train_dataset = train_dataset.select(range(1000))
    tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
    context_length = 128

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}
    tokenized_dataset = train_dataset.map(tokenize, batched=True, remove_columns=['text',  'label'])
    print(f'Training set size: {len(tokenized_dataset)}')
    batch_size = 64
    print(f'Total number of training steps: {len(tokenized_dataset)//batch_size}')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    # samples = [tokenized_dataset[i] for i in range(5)]
    # samples_out = data_collator(tokenized_dataset[:5])

    model = GPT2LMHeadModel.from_pretrained('gpt2-large', cache_dir=f'{constants.ROOT}/.cache/')

    print(f'Model size: {utils.count_parameters(model)}')

    training_args = TrainingArguments(
        output_dir=f"{constants.ROOT}/models/gpt2-imdb/", #The output directory
        overwrite_output_dir=True, #overwrite the content of the output directory
        num_train_epochs=1, # number of training epochs
        per_device_train_batch_size=batch_size, # batch size for training
        per_device_eval_batch_size=batch_size,  # batch size for evaluation
        eval_steps = 100, # Number of update steps between two evaluations.
        save_steps = 100, # after # steps model is saved
        warmup_steps=500,# number of warmup steps for learning rate scheduler
        report_to='wandb',
        logging_steps=1,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset,
        # eval_dataset=test_dataset,
    )

    trainer.train()



if __name__ == "__main__":
    main()

