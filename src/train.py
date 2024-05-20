import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


def split_data(df: pd.DataFrame):
    """
    Given a DataFrame, it creates the X and Y frames, and splits the Data.
    """
    df.dropna(inplace=True)
    X = df.drop(columns=['Result'])
    Y = df['Result']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.2, random_state=42)
    return x_train, y_train, x_test, y_test


def train(df: pd.DataFrame, model):
    x_train, y_train, x_test, y_test = split_data(df=df)
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
