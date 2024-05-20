import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


def split_data(train_df: pd.DataFrame, test_df:pd.DataFrame):
    """
    Given a DataFrame, it creates the X and Y frames, and splits the Data.
    """
    train_df.dropna(inplace=True)
    x_train = train_df.drop(columns=['Result'])
    y_train = train_df['Result']

    test_df.dropna(inplace=True)
    x_test = test_df.drop(columns=['Result'])
    y_test = test_df['Result']

    return x_train, y_train, x_test, y_test


def train_test(train_df: pd.DataFrame, model, test_df: pd.DataFrame):
    x_train, y_train, x_test, y_test = split_data(train_df=train_df, test_df=test_df)
    # Train Model
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.plot()
    plt.savefig("cm.png")

    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    print(f"Accuracy Score: {accuracy}")
    return
