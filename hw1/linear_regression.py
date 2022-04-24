import numpy as np
import sklearn
from pandas import DataFrame
from typing import List
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.utils import check_array
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.utils.validation import check_X_y, check_is_fitted


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        check_is_fitted(self, "weights_")

        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = X.dot(self.weights_)
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """

        X, y = check_X_y(X, y)

        # TODO:
        #  Calculate the optimal weights using the closed-form solution you derived.
        #  Use only numpy functions. Don't forget regularization!

        w_opt = None
        # ====== YOUR CODE: ======
        reg=X.shape[0]*self.reg_lambda*np.identity(X.shape[1])
        reg[0,0]=0
        w_opt = np.linalg.inv(X.T.dot(X) + reg).dot(X.T.dot(y))
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


def fit_predict_dataframe(
    model, df: DataFrame, target_name: str, feature_names: List[str] = None,
):
    """
    Calculates model predictions on a dataframe, optionally with only a subset of
    the features (columns).
    :param model: An sklearn model. Must implement fit_predict().
    :param df: A dataframe. Columns are assumed to be features. One of the columns
        should be the target variable.
    :param target_name: Name of target variable.
    :param feature_names: Names of features to use. Can be None, in which case all
        features are used.
    :return: A vector of predictions, y_pred.
    """
    # TODO: Implement according to the docstring description.
    # ====== YOUR CODE: ======
    try:
        if not feature_names:
            y_pred = model.fit_predict(df.iloc[:, :-1], df.loc[:, target_name])
        else:
            y_pred = model.fit_predict(df.loc[:, feature_names], df.loc[:, target_name])
    except:
        y_pred = model.fit_predict(df.loc[:, feature_names], df.loc[:, target_name])

    # ========================
    return y_pred


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X: np.ndarray):
        """
        :param X: A tensor of shape (N,D) where N is the batch size and D is
        the number of features.
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """
        X = check_array(X, ensure_2d=True)

        # TODO:
        #  Add bias term to X as the first feature.
        #  See np.hstack().

        xb = None
        # ====== YOUR CODE: ======
        xb = np.hstack((np.ones((len(X[:,0]),1), dtype=int), X))
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======

        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)

        # TODO:
        #  Transform the features of X into new features in X_transformed
        #  Note: You CAN count on the order of features in the Boston dataset
        #  (this class is "Boston-specific"). For example X[:,1] is the second
        #  feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        poly = PolynomialFeatures(self.degree)
        X_poly = poly.fit_transform(X)
        X_transformed = np.delete(X_poly, [0, 1, 3, 11], axis=1)

        # ========================

        return X_transformed


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it

    # ====== YOUR CODE: ======

    corr = df.corr(method='pearson')[[target_feature]].abs()
    corr.drop([target_feature], axis=0, inplace=True)
    top_n_corr = corr.nlargest(5,target_feature)[target_feature].unique()
    top_n_features = corr.nlargest(5,target_feature)[target_feature].index
    # ========================

    return top_n_features, top_n_corr


def mse_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes Mean Squared Error.
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: MSE score.
    """

    # TODO: Implement MSE using numpy.
    # ====== YOUR CODE: ======
    mse = ((y - y_pred)**2).mean(axis=0)
    # ========================
    return mse


def r2_score(y: np.ndarray, y_pred: np.ndarray):
    """
    Computes R^2 score,
    :param y: Predictions, shape (N,)
    :param y_pred: Ground truth labels, shape (N,)
    :return: R^2 score.
    """

    # TODO: Implement R^2 using numpy.
    # ====== YOUR CODE: ======
    ss_tot = ((y - y.mean()) ** 2).sum()
    ss_res = ((y - y_pred) ** 2).sum()
    r2 = 1 - (ss_res / ss_tot)
 # ========================
    return r2


def cv_best_hyperparams(
    model: BaseEstimator, X, y, k_folds, degree_range, lambda_range
):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #  Notes:
    #  - You can implement it yourself or use the built in sklearn utilities
    #    (recommended). See the docs for the sklearn.model_selection package
    #    http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    #  - If your model has more hyperparameters (not just lambda and degree)
    #    you should add them to the search.
    #  - Use get_params() on your model to see what hyperparameters is has
    #    and their names. The parameters dict you return should use the same
    #    names as keys.
    #  - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======
    param_grid = {'bostonfeaturestransformer__degree': degree_range,
                  'linearregressor__reg_lambda': lambda_range}
    R2 = sklearn.metrics.make_scorer(r2_score)
    grid_search = sklearn.model_selection.GridSearchCV(estimator = model,
                                                       param_grid = param_grid,
                                                       cv = k_folds,
                                                       scoring = R2)
    best_params = grid_search.fit(X,y).best_params_
    # ========================

    return best_params

import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import unittest


plt.rcParams.update({'font.size': 14})
np.random.seed(42)
test = unittest.TestCase()

import sklearn.datasets
import warnings

# Load data we'll work with - Boston housing dataset
# We'll use sklearn's built-in data
with warnings.catch_warnings():
    warnings.simplefilter("ignore")

    ds_boston = sklearn.datasets.load_boston()

feature_names = list(ds_boston.feature_names)

n_features = len(feature_names)
x, y = ds_boston.data, ds_boston.target
n_samples = len(y)
print(f'Loaded {n_samples} samples')

# Load into a pandas dataframe and show some samples
df_boston = pd.DataFrame(data=x, columns=ds_boston.feature_names)
df_boston = df_boston.assign(MEDV=y)
df_boston.head(10).style.background_gradient(subset=['MEDV'], high=1.)

n_top_features = 5
top_feature_names, top_corr = top_correlated_features(df_boston, 'MEDV', n_top_features)

def evaluate_accuracy(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates mean squared error (MSE) and coefficient of determination (R-squared).
    :param y: Target values.
    :param y_pred: Predicted values.
    :return: A tuple containing the MSE and R-squared values.
    """
    mse = mse_score(y, y_pred)
    rsq = r2_score(y, y_pred)
    return mse, rsq

from sklearn.metrics import r2_score as r2, mean_squared_error as mse

import sklearn.preprocessing
import sklearn.pipeline

# Create our model as a pipline:
# First we scale each feature, then the bias trick is applied, then the regressor
model = sklearn.pipeline.make_pipeline(
    sklearn.preprocessing.StandardScaler(),
    BiasTrickTransformer(),
    LinearRegressor(),
)


fig, ax = plt.subplots(nrows=1, ncols=n_top_features, sharey=True, figsize=(20, 5))
actual_mse = []

# Fit a single feature at a time
for i, feature_name in enumerate(top_feature_names):
    y_pred = fit_predict_dataframe(model, df_boston, 'MEDV', [feature_name])
    mse, rsq = evaluate_accuracy(y, y_pred)

    # Plot
    xf = df_boston[feature_name].values.reshape(-1, 1)
    x_line = np.arange(xf.min(), xf.max(), 0.1, dtype=float).reshape(-1, 1)
    y_line = model.predict(x_line)
    ax[i].scatter(xf, y, marker='o', edgecolor='black')
    ax[i].plot(x_line, y_line, color='red', lw=2, label=f'fit, $R^2={rsq:.2f}$')
    ax[i].set_ylabel('MEDV')
    ax[i].set_xlabel(feature_name)
    ax[i].legend(loc='upper right')

    actual_mse.append(mse)

# Test regressor implementation
print(actual_mse)
expected_mse = [38.862, 43.937, 62.832, 64.829, 66.040]
for i in range(len(expected_mse)):
    test.assertAlmostEqual(expected_mse[i], actual_mse[i], delta=1e-1)