import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from data_preparation import prepare_regular_season_csv, prepare_test_data
from train import train_test, train


def pipeline(config):
    if config.get("ingestion").get("active"):
        df = prepare_regular_season_csv(config.get("regularSeason"))
        df.to_csv("../data/MMedianSeasonAggStreak_2.csv", index=False)
    else:
        df = pd.read_csv(config.get("ingestion").get("existingFile"))

    test_df = prepare_test_data(config.get("testData"), train_df=df)
    # train_test(train_df=df, test_df=test_df)
    result = train(train_df=df, test_df=test_df)


if __name__ == "__main__":
    config_name = "config.json"
    with open(config_name, "r") as file:
        config = json.load(file)
    pipeline(config=config)
