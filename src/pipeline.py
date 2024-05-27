import json
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from data_preparation import prepare_regular_season_csv, prepare_test_data
from train import train_test, train


def pipeline(config):
    if config.get("male").get("active"):
        if config.get("male", None).get("ingestion").get("active"):
            df = prepare_regular_season_csv(config.get("male", None).get("regularSeason"))
            df.to_csv(config.get("male").get("ingestion").get("outputCSV"), index=False)
        else:
            df = pd.read_csv(config.get("male").get("ingestion").get("existingFile"))

        test_df = prepare_test_data(config.get("male").get("testData"), train_df=df)
        # train_test(train_df=df, test_df=test_df)
        model = train(train_df=df, test_df=test_df, best_model=config.get("male").get("model"), type="male")

    if config.get("female").get("active"):
        if config.get("female", None).get("ingestion").get("active"):
            df = prepare_regular_season_csv(config.get("female", None).get("regularSeason"))
            df.to_csv(config.get("female").get("ingestion").get("outputCSV"), index=False)
        else:
            df = pd.read_csv(config.get("female").get("ingestion").get("existingFile"))

        test_df = prepare_test_data(config.get("female").get("testData"), train_df=df)
        # train_test(train_df=df, test_df=test_df)
        model = train(train_df=df, test_df=test_df, best_model=config.get("female").get("model"), type="female")



if __name__ == "__main__":
    config_name = "config.json"
    with open(config_name, "r") as file:
        config = json.load(file)
    pipeline(config=config)
