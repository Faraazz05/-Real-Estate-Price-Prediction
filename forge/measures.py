import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def correlation_heatmap(df: pd.DataFrame, figsize=(8,6)):
    corr = df.corr()
    plt.figure(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()


def residual_plot(y_true, y_pred, figsize=(8,6)):
    residuals = y_true - y_pred
    plt.figure(figsize=figsize)
    sns.scatterplot(x=y_pred, y=residuals, alpha=0.6)
    plt.axhline(0, color="red", linestyle="--")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()


def regression_metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"rmse": rmse, "r2": r2}
