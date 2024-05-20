import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from data_preparation import prepare_regular_season_csv
from train import train


def pipeline(config):
    if config.get("ingestion").get("active"):
        df = prepare_regular_season_csv(config.get("regularSeason"))
    else:
        df = pd.read_csv(config.get("ingestion").get("existingFile"))
    model = LogisticRegression(max_iter=10_000, C=1000_000)
    train(df=df, model=model)


if __name__ == "__main__":
    config_name = "config.json"
    with open(config_name, "r") as file:
        config = json.load(file)
    pipeline(config=config)
