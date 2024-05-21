import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pickle


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
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)

    cm = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.plot()
    plt.savefig("plots/cm_train.png")

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.plot()
    plt.savefig("plots/cm_test.png")

    accuracy = accuracy_score(y_pred=y_train_pred, y_true=y_train)
    print(f"Accuracy Score Train: {accuracy}")
    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    print(f"Accuracy Score Test: {accuracy}")
    pickle.dump(model, open("model/model.sav", "wb"))
    find_best_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    return


def find_best_model(x_train, y_train, x_test, y_test):

    # Definir los modelos a probar
    modelos = [
        # ('SVC', SVC()),
        ('RandomForest', RandomForestClassifier()),
        ('LogisticRegression', LogisticRegression()),
        ('XGBoost', XGBClassifier()),
        ('GradientBoosting', GradientBoostingClassifier())
    ]

    # Definir los hiperparámetros para cada modelo
    hiperparametros = [
        # {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
        {'n_estimators': [10, 50, 100, 200], 'max_features': ['sqrt', 'log2']},
        {'C': [0.1, 1, 10, 100, 1_000, 10_000], 'penalty': ['l1', 'l2'], "max_iter": [100, 1_000, 10_000]},
        {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]},
        {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]}
    ]

    # Inicializar la variable para almacenar el mejor modelo y su precisión
    mejor_modelo = None
    mejor_precision = 0

    # Probar cada modelo con cada combinación de hiperparámetros
    for i in range(len(modelos)):
        grid = GridSearchCV(estimator=modelos[i][1], param_grid=hiperparametros[i], scoring='accuracy', n_jobs=4)
        grid.fit(x_train, y_train)
        modelo = grid.best_estimator_
        precision = accuracy_score(y_test, modelo.predict(x_test))
        print(f"Para el modelo {i} best accuracy: {precision}")
        # Si la precisión del modelo actual es mayor que la del mejor modelo hasta ahora, actualizar el mejor modelo
        if precision > mejor_precision:
            mejor_modelo = modelo
            mejor_precision = precision

    # Imprimir el mejor modelo y su precisión
    print(f'Mejor modelo: {mejor_modelo}')
    print(f'Precisión del mejor modelo: {mejor_precision}')
