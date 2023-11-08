from typing import Any

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns


def load_data_frame() -> pd.DataFrame:
    """
    Laad een DataFrame vanuit een CSV-bestand met gegevens en verwijder duplicaten.

    Returns:
        pd.DataFrame: Het ingelezen DataFrame met de gegevens, waarbij duplicaten zijn verwijderd.
    """
    return pd.read_csv('winequality-red.csv').drop_duplicates()


def show_bar_chart(df: pd.DataFrame) -> None:
    """
    Toon een staafdiagram met de gemiddelde waarden per label in een DataFrame.

    Parameters:
        df (DataFrame): Het DataFrame met gegevens waarvan de gemiddelde waarden worden weergegeven.

    Returns:
        None
    """
    labels = df.columns
    mean_values = df.describe().loc['mean']

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels, mean_values)

    ax.set_title('Gemiddelde waarden per label')
    ax.set_xlabel('Label')
    ax.set_ylabel('Gemiddelde waarde')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()


def sum_missing_values(df: pd.DataFrame) -> None:
    """
    Toon het aantal ontbrekende waarden in elke kolom van een DataFrame.

    Parameters:
        df (DataFrame): Het DataFrame waarin ontbrekende waarden worden geteld.

    Returns:
        None
    """
    print(df.isnull().sum())


def split_train_test(df: pd.DataFrame) -> Any:
    """
    Splits de gegevens in een DataFrame in trainings- en testsets.

    Parameters:
        df (DataFrame): Het DataFrame met gegevens om te splitsen.

    Returns:
        Tuple: Een tuple met de trainings- en testdatasets (X_train, X_test, y_train, y_test).
    """
    X = df.drop('quality', axis=1)
    y = df['quality']
    return train_test_split(X, y, test_size=0.2, random_state=42)


def train_linear_regression(X_train, y_train, fit_intercept) -> LinearRegression:
    """
    Traint een lineaire regressiemodel.

    Parameters:
        X_train (pd.DataFrame): De trainingskenmerken.
        y_train (pd.Series): De trainingslabels.
        fit_intercept (bool): Of het model een intercept moet bevatten.

    Returns:
        LinearRegression: Het getrainde lineaire regressiemodel.
    """
    # Fit intercept is  verantwoordelijk voor het vastleggen van de verticale verschuiving
    # van de regressielijn en het aanpassen van het model aan de basislijn van de gegevens (kruispunt).
    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X_train, y_train)
    return model


def test_linear_regression(X_train, X_test, y_train, y_test, fit_intercept=True, normalize=False) -> dict[
    str, float | Any]:
    """
    Evalueert een lineaire regressiemodel op de testgegevens.

    Parameters:
        X_train (pd.DataFrame): De trainingskenmerken.
        X_test (pd.DataFrame): De testkenmerken.
        y_train (pd.Series): De trainingslabels.
        y_test (pd.Series): De testlabels.
        fit_intercept (bool, optional): Of het model een intercept moet bevatten. Standaard True.
        normalize (bool, optional): Of de gegevens moeten worden genormaliseerd. Standaard False.

    Returns:
        dict: Een dictionary met de resultaten, inclusief Mean Squared Error (MSE) en R-squared (R2) scores.
    """
    results = {}

    # De schaal van features in een dataset gelijk te trekken.
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    model = train_linear_regression(X_train, y_train, fit_intercept)
    y_pred = model.predict(X_test)

    results['mse'] = mean_squared_error(y_test, y_pred)
    results['r2'] = r2_score(y_test, y_pred)

    return results


def compare_linear_regression_models(X_train, X_test, y_train, y_test):
    """
    Vergelijkt lineaire regressiemodellen met verschillende hyperparameters.

    Parameters:
        X_train (pd.DataFrame): De trainingskenmerken.
        X_test (pd.DataFrame): De testkenmerken.
        y_train (pd.Series): De trainingslabels.
        y_test (pd.Series): De testlabels.

    Returns:
        dict: Een dictionary met de resultaten van verschillende lineaire regressiemodellen.
    """
    results = {}

    model1 = test_linear_regression(X_train, X_test, y_train, y_test, fit_intercept=False, normalize=False)
    results['Model 1: fit_intercept=False, normalize=False'] = model1

    model2 = test_linear_regression(X_train, X_test, y_train, y_test, fit_intercept=False, normalize=True)
    results['Model 2: fit_intercept=False, normalize=True)'] = model2

    model3 = test_linear_regression(X_train, X_test, y_train, y_test, fit_intercept=True, normalize=False)
    results['Model 3: fit_intercept=True, normalize=False'] = model3

    model4 = test_linear_regression(X_train, X_test, y_train, y_test, fit_intercept=True, normalize=True)
    results['Model 4: fit_intercept=True, normalize=True'] = model4

    return results


def plot_linear_regression_results(results):
    """
    Plot de resultaten van lineaire regressiemodellen met verschillende hyperparameters.

    Parameters:
        results (dict): Een dictionary met de resultaten van verschillende lineaire regressiemodellen.

    Returns:
        None
    """
    model_names = results.keys()
    mse_scores = [model['mse'] for model in results.values()]
    r2_scores = [model['r2'] for model in results.values()]

    plt.figure(figsize=(10, 6))
    plt.bar(model_names, mse_scores, label='Mean Squared Error (MSE)', alpha=0.7)
    plt.bar(model_names, r2_scores, label='R-squared (R2)', alpha=0.7)

    plt.xlabel('Models')
    plt.ylabel('Scores')
    plt.title('Linear Regression Model Performance with Different Hyperparameters')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()


def train_logistic_regression(X_train, y_train, C, max_iter) -> LogisticRegression:
    """
    Traint een logistisch regressiemodel.

    Parameters:
        X_train (pd.DataFrame): De trainingskenmerken.
        y_train (pd.Series): De trainingslabels.
        C (float): De inverse regularisatiesterkte.
        max_iter (int): Het maximale aantal iteraties.

    Returns:
        LogisticRegression: Het getrainde logistische regressiemodel.
    """
    model = LogisticRegression(C=C, max_iter=max_iter)
    model.fit(X_train, y_train)
    return model


def test_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, max_iter=100) -> dict[str, Any]:
    """
    Evalueert een logistisch regressiemodel op de testgegevens.

    Parameters:
        X_train (pd.DataFrame): De trainingskenmerken.
        X_test (pd.DataFrame): De testkenmerken.
        y_train (pd.Series): De trainingslabels.
        y_test (pd.Series): De testlabels.
        C (float, optional): De inverse regularisatiesterkte. Standaard 1.0.
        max_iter (int, optional): Het maximale aantal iteraties. Standaard 100.

    Returns:
        dict: Een dictionary met de resultaten, inclusief de confusion matrix, F1 Score, Precision en Accuracy scores.
    """
    results = {}

    model = train_logistic_regression(X_train, y_train, C, max_iter)
    y_pred = model.predict(X_test)

    results['confusion_matrix'] = conf_matrix(y_test, y_pred)
    results['f1_score'] = f1_score(y_test, y_pred, average='weighted')
    results['precision'] = precision_score(y_test, y_pred, average='weighted', zero_division=1)
    results['accuracy'] = accuracy_score(y_test, y_pred)

    return results


def compare_logistic_regression_models(X_train, X_test, y_train, y_test):
    """
    Vergelijkt logistische regressiemodellen met verschillende hyperparameters.

    Parameters:
        X_train (pd.DataFrame): De trainingskenmerken.
        X_test (pd.DataFrame): De testkenmerken.
        y_train (pd.Series): De trainingslabels.
        y_test (pd.Series): De testlabels.

    Returns:
        dict: Een dictionary met de resultaten van verschillende logistische regressiemodellen.
    """
    results = {}

    model1 = test_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, max_iter=10000)
    results['Model 1: C=1.0, max_iter=100'] = model1

    model2 = test_logistic_regression(X_train, X_test, y_train, y_test, C=0.1, max_iter=10000)
    results['Model 2: C=0.1, max_iter=100'] = model2

    model3 = test_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, max_iter=100000)
    results['Model 3: C=1.0, max_iter=1000'] = model3

    model4 = test_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, max_iter=50000)
    results['Model 4: C=1.0, max_iter=500'] = model4

    model5 = test_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, max_iter=10000)
    results['Model 5: C=1.0, max_iter=100'] = model5

    model6 = test_logistic_regression(X_train, X_test, y_train, y_test, C=1.0, max_iter=10000)
    results['Model 6: C=1.0, max_iter=100'] = model6

    return results


def conf_matrix(y_test, y_pred) -> Any:
    """
    Genereert en toont de Confusion Matrix voor de voorspelde resultaten.

    Parameters:
    y_test (array-like): De werkelijke labels.
    y_pred (array-like): De voorspelde labels.

    Returns:
    array: De Confusion Matrix in de vorm van een array.

    Deze functie berekent de Confusion Matrix op basis van de werkelijke labels (y_test) en de voorspelde labels (y_pred).
    Vervolgens wordt de matrix gevisualiseerd met behulp van een heatmap en getoond in een grafiek.

    """
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Voorspelde labels')
    plt.ylabel('Echte labels')
    plt.title('Confusion Matrix')
    plt.show()

    return cm

def plot_logistic_regression_results(results):
    """
    Plot de prestaties van modellen met verschillende hyperparameters.

    Parameters:
    results (dict): Een dictionary met hyperparameters als sleutels en prestatiegegevens als waarden.

    Deze functie maakt een staafdiagram om de prestaties van verschillende modellen weer te geven op basis van
    verschillende hyperparameters. De prestatiegegevens omvatten F1-score, precisie en nauwkeurigheid.
    """
    hyperparams = results.keys()
    f1_scores = [model['f1_score'] for model in results.values()]
    precision_scores = [model['precision'] for model in results.values()]
    accuracy_scores = [model['accuracy'] for model in results.values()]

    plt.figure(figsize=(10, 10))
    plt.bar(hyperparams, f1_scores, label='F1 Score', alpha=0.7)
    plt.bar(hyperparams, precision_scores, label='Precisie', alpha=0.7)
    plt.bar(hyperparams, accuracy_scores, label='Nauwkeurigheid', alpha=0.7)

    plt.xlabel('Hyperparameters')
    plt.ylabel('Scores')
    plt.title('Modelprestaties met verschillende Hyperparameters')
    plt.legend()
    plt.xticks(rotation=45)
    plt.show()

# Todo:
# grafieken (maakt niet uit wat) confusion matrix
# vergelijken van hyper params, met grafieken.


if __name__ == '__main__':
    data_frame = load_data_frame()

    show_bar_chart(data_frame)

    X_train, X_test, y_train, y_test = split_train_test(data_frame)

    model_results = compare_linear_regression_models(X_train, X_test, y_train, y_test)

    plot_linear_regression_results(model_results)

    for model_name, model in model_results.items():
        print(f"Model: {model_name}")
        print(f'MSE: {model["mse"]}')
        print(f'R^2: {model["r2"]}')
        print('____________________________')

    model_results = compare_logistic_regression_models(X_train, X_test, y_train, y_test)

    plot_logistic_regression_results(model_results)

    for model_name, model in model_results.items():
        print(f"Model: {model_name}")
        print("Confusion Matrix:")
        print(model["confusion_matrix"])
        print(f'F1 Score: {model["f1_score"]}')
        print(f'Precision: {model["precision"]}')
        print(f'Accuracy: {model["accuracy"]}')
        print('____________________________')