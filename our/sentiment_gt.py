from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, pipeline, AutoModelForSequenceClassification

from . import constants


class SentimentGT:
    def __init__(self):
        sentiment_model = AutoModelForSequenceClassification.from_pretrained(
                pretrained_model_name_or_path="siebert/sentiment-roberta-large-english",
                cache_dir=f'{constants.ROOT}/.cache/')
        tokenizer = AutoTokenizer.from_pretrained("siebert/sentiment-roberta-large-english",
                cache_dir=f'{constants.ROOT}/.cache/')
        self.sentiment_pipeline = pipeline("sentiment-analysis", model=sentiment_model,
                tokenizer=tokenizer, device='cuda')

        self.sent_kwargs = {"top_k": None, "function_to_apply": "softmax",
                "batch_size": 128}


    def get_score(self, texts):
        outputs = self.sentiment_pipeline(texts, **self.sent_kwargs)
        assert all([len(item)==2 for item in outputs])
        scores = [item[0]['score'] if item[0]['label']=='POSITIVE' else
                item[1]['score'] for item in outputs]
        return scores

