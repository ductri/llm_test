from transformers import pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification


from our import constants


if __name__ == "__main__":
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path="siebert/sentiment-roberta-large-english",
            cache_dir=f'{constants.ROOT}/.cache/')
    tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english")
    sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model,
            tokenizer=tokenizer)
    data = ['First one was alright,If you do not like the Spanish Quatermain-Ghezzi',
             'First one was great.The second is OK.Maybe it has a small part like flashbacks and',
             "First one was just beyond enjoyment. Second was a movie that should sure be kicked em' was",
             ]

    outputs = sentiment_pipeline(data)
    print(outputs)
    __import__('pdb').set_trace()
    print()

