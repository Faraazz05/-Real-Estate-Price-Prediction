import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class RealEstateModel:
    """
    Multiple Linear Regression model for real estate price prediction.
    """

    def __init__(self):
        self.model = LinearRegression()
        self.is_trained = False


    def predict(self, X):
        return self.model.predict(X)

    def get_coefficients(self, feature_names):
        return pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": self.model.coef_
        })

    def train(self, X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42):
        """
        Train the linear regression model on given data.

        Parameters
        ----------
        X : pd.DataFrame
            Features.
        y : pd.Series
            Target variable (price).
        test_size : float
            Fraction of data to use for testing.
        random_state : int
            Random seed for reproducibility.

        Returns
        -------
        dict
            Training and testing metrics (RMSE, R²).
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        self.model.fit(X_train, y_train)
        self.is_trained = True

        # Predictions
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)

        # Metrics
        metrics = {
            "train_rmse": np.sqrt(mean_squared_error(y_train, y_train_pred)),
            "train_r2": r2_score(y_train, y_train_pred),
            "test_rmse": np.sqrt(mean_squared_error(y_test, y_test_pred)),
            "test_r2": r2_score(y_test, y_test_pred),
        }

        return metrics
    


    def get_coefficients(self, feature_names: list) -> pd.DataFrame:
        """
        Get model coefficients with feature names.

        Returns
        -------
        pd.DataFrame
            Coefficients table.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")

        coefs = pd.DataFrame({
            "feature": feature_names,
            "coefficient": self.model.coef_
        })
        coefs.loc[len(coefs)] = ["intercept", self.model.intercept_]
        return coefs

    def predict(self, X_new: pd.DataFrame) -> np.ndarray:
        """
        Predict price for new input data.

        Parameters
        ----------
        X_new : pd.DataFrame
            New feature data.

        Returns
        -------
        np.ndarray
            Predicted prices.
        """
        if not self.is_trained:
            raise ValueError("Model is not trained yet.")

        return self.model.predict(X_new)
    
    def train(self, X, y):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = LinearRegression()
        self.model.fit(X_train, y_train)

        y_pred = self.model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        return {
            "mse": mse,
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

    def predict(self, X):
        return self.model.predict(X)

    def get_coefficients(self, feature_names):
        return pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": self.model.coef_
        })
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
