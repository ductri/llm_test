import pandas as pd

from our import constants


if __name__ == "__main__":
    df = pd.read_csv(f'{constants.ROOT}/data/sentiment_imdb_preference_dataset.csv')

