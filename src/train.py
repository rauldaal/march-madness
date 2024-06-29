import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score, recall_score, f1_score, roc_auc_score, log_loss, roc_curve

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
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


def evaluate_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)

    cm_train = confusion_matrix(y_true=y_train, y_pred=y_train_pred)

    cm_test = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # Compute Train Metrics
    accuracy_train = accuracy_score(y_pred=y_train_pred, y_true=y_train)
    # Recall
    recall_train = recall_score(y_train, y_train_pred)
    # Puntuación F1
    f1_train = f1_score(y_train, y_train_pred)
    # Curva ROC (AUC)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, y_train_pred)
    auc_train = roc_auc_score(y_train, y_train_pred)
    # Pérdida logarítmica
    loss_train = log_loss(y_train, y_train_pred)

    # Compute Test Metrics
    accuracy_test = accuracy_score(y_pred=y_pred, y_true=y_test)
    # Recall
    recall_test = recall_score(y_test, y_pred)
    # Puntuación F1
    f1_test = f1_score(y_test, y_pred)
    # Curva ROC (AUC)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_pred)
    auc_test = roc_auc_score(y_test, y_pred)
    # Pérdida logarítmica
    loss_test = log_loss(y_test, y_pred)

    return ([accuracy_train, recall_train, f1_train, [fpr_train, tpr_train, thresholds_train], auc_train, loss_train, cm_train],
            [accuracy_test, recall_test, f1_test, [fpr_test, tpr_test, thresholds_test], auc_test, loss_test, cm_test])


def custom_train(train_df: pd.DataFrame, test_df: pd.DataFrame, model_instance, params):
    x_train, y_train, x_test, y_test = split_data(train_df=train_df, test_df=test_df)
    grid = get_grid(params=params)
    best_acc = 0
    best_params = None
    best_all_results = None
    for i, g in enumerate(grid):
        try:
            model = model_instance(**g)
            print(f"Evaluando Modelo {str(model)}: {i}/{len(grid)}")
            results = evaluate_model(model=model, x_test=x_test, x_train=x_train, y_test=y_test, y_train=y_train)
            if results[1][0] > best_acc:
                best_acc = results[1][0]
                best_params = g
                best_all_results = results
        except Exception as e:
            print(e)

    return best_acc, best_params, best_all_results


def create_comparision_graphic(results, type):
    categorias = ['Accuracy Train', 'Recall Train', 'F1 Score Train', 'Accuracy Test', 'Recall Test', 'F1 Score Test']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    N = len(categorias)
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for i, (model, values) in enumerate(results.items()):
        angulos = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
        grpah_metrics = [values[2][0][0], values[2][0][1], values[2][0][2], values[2][1][0], values[2][1][1], values[2][1][2]]
        grpah_metrics += grpah_metrics[:1]
        angulos += angulos[:1]
        ax.fill(angulos, grpah_metrics, color=colors[i], alpha=0.25)
        ax.plot(angulos, grpah_metrics, color=colors[i], linewidth=2, linestyle='solid', label=model)
        ax.set_xticks(angulos[:-1])
        ax.set_xticklabels(categorias)
        ax.set_rscale('linear')
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.plot()
    plt.savefig(f"plots/{type}/_spider_graph.png")
    for i, (model, values) in enumerate(results.items()):
        ConfusionMatrixDisplay(confusion_matrix=values[2][0][-1]).plot()
        plt.plot()
        plt.savefig(f"plots/{type}/2024_cm_{model}_train.png")
        ConfusionMatrixDisplay(confusion_matrix=values[2][1][-1]).plot()
        plt.plot()
        plt.savefig(f"plots/{type}/2024_cm_{model}_test.png")
    plt.close('all')
    return


def train(train_df: pd.DataFrame, test_df: pd.DataFrame, type: str, best_model=None):
    modelos = [
        # ('SVC', SVC()),
        ('XGBoost', XGBClassifier),
        ('RandomForest', RandomForestClassifier),
        ('LogisticRegression', LogisticRegression),
        ('GradientBoosting', GradientBoostingClassifier),
        ('GaussianNB', GaussianNB)
    ]
    hiperparametros = [
        # {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
        {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]},
        {'n_estimators': [10, 50, 100, 200], 'max_features': ['sqrt', 'log2']},
        {'C': [0.1, 1, 10, 100, 1_000, 10_000, 100_000], 'penalty': ['l1', 'l2'], "max_iter": [100, 1_000, 10_000], "n_jobs": [-1]},
        {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]},
        {'var_smoothing': [1e-10, 1e-9, 1e-8, 1e-5, 1e-3]}
    ]

    result = {}

    for m, h in zip(modelos, hiperparametros):
        acc, params, all_results = custom_train(train_df=train_df, test_df=test_df, model_instance=m[1], params=h)
        result[m[0]] = (acc, params, all_results)

    print(result)
    create_comparision_graphic(results=result, type=type)
    model_instance = None
    for name, instance in modelos:
        if name == best_model:
            model_instance = instance
    params = result.get(best_model)[1]

    model = train_with_params(model=model_instance, params=params, train_df=train_df, test_df=test_df)
    return model


def train_with_params(model, params, train_df, test_df):
    x_train, y_train, x_test, y_test = split_data(train_df=train_df, test_df=test_df)
    model = model(**params)
    model.fit(x_train, y_train)
    pickle.dump(model, open("model/model.sav", "wb"))
    return model


def get_grid(params):
    valores_parametros = list(params.values())
    parametros_claves = list(params.keys())  # ['n_estimators', 'learning_rate']
    combinations = itertools.product(*valores_parametros)
    res = list(combinations)
    result = []
    for r in res:
        dic = {}
        for k, v in zip(parametros_claves, r):
            dic[k] = v
        result.append(dic)
    return result


def train_test(train_df: pd.DataFrame, test_df: pd.DataFrame):
    x_train, y_train, x_test, y_test = split_data(train_df=train_df, test_df=test_df)
    # Train Model
    model = find_best_model(x_train=x_train, x_test=x_test, y_train=y_train, y_test=y_test)

    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_pred = model.predict(x_test)
    cm = confusion_matrix(y_true=y_train, y_pred=y_train_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.plot()
    plt.savefig("plots/2024_cm_train.png")

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    ConfusionMatrixDisplay(confusion_matrix=cm).plot()
    plt.plot()
    plt.savefig("plots/2024_cm_test.png")

    accuracy = accuracy_score(y_pred=y_pred, y_true=y_test)
    print(f"Accuracy Score Test: {accuracy}")
    accuracy = accuracy_score(y_pred=y_train_pred, y_true=y_train)
    print(f"Accuracy Score Train: {accuracy}")

    recall = recall_score(y_test, y_pred)

    # Puntuación F1
    f1 = f1_score(y_test, y_pred)

    # Curva ROC (AUC)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)

    # Pérdida logarítmica
    loss = log_loss(y_test, y_pred)

    pickle.dump(model, open("model/model.sav", "wb"))

    return


def find_best_model(x_train, y_train, x_test, y_test):

    # Definir los modelos a probar
    modelos = [
        # ('SVC', SVC()),
        ('XGBoost', XGBClassifier()),
        ('RandomForest', RandomForestClassifier()),
        ('LogisticRegression', LogisticRegression()),
        ('GradientBoosting', GradientBoostingClassifier())
    ]

    # Definir los hiperparámetros para cada modelo
    hiperparametros = [
        # {'C': [0.1, 1, 10, 100], 'kernel': ['linear', 'rbf']},
        {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2, 0.3]},
        {'n_estimators': [10, 50, 100, 200], 'max_features': ['sqrt', 'log2']},
        {'C': [0.1, 1, 10, 100, 1_000, 10_000, 100_000], 'penalty': ['l1', 'l2'], "max_iter": [100, 1_000, 10_000]},
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
        accuracy = accuracy_score(y_test, modelo.predict(x_test))
        print(f"Para el modelo {modelos[i][0]} best accuracy: {accuracy}")
        # Si la precisión del modelo actual es mayor que la del mejor modelo hasta ahora, actualizar el mejor modelo
        if accuracy > mejor_precision:
            mejor_modelo = modelo
            mejor_precision = accuracy

    # Imprimir el mejor modelo y su precisión
    print(f'Mejor modelo: {mejor_modelo}')
    print(f'Precisión del mejor modelo: {mejor_precision}')
    return mejor_modelo
