import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.metrics import f1_score, mean_squared_error, jaccard_score
from scipy.spatial.distance import cosine
from dython.nominal import compute_associations


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(np.subtract(y_true, y_pred)))


def euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(np.sum(np.power(np.subtract(y_true, y_pred), 2)))


def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    return np.sum(y_true * y_pred) / (np.sqrt(np.sum(y_true ** 2)) * np.sqrt(np.sum(y_pred ** 2)))


class TableEvaluator:
    """

    """
    def __init__(
            self,
            real_data: pd.DataFrame,
            fake_data: pd.DataFrame,
            unique_threshold=0,
            metric='pearsonr',
            verbose=False,
            name: str = None,
            seed=59
    ):
        self.name = name
        self.unique_thresh = unique_threshold
        self.real_data = real_data.copy()
        self.fake = fake_data.copy()
        self.comparison_metric = getattr(stats, metric)
        self.verbose = verbose
        self.random_seed = seed

    def correlation_distance(self, how: str = 'euclidean') -> float:
        if how == 'euclidean':
            distance_func = euclidean_distance
        elif how == 'mae':
            distance_func = mean_absolute_error
        elif how == 'rmse':
            distance_func = rmse
        elif how == 'cosine':
            def custom_cosine(a, b):
                return cosine(a.reshape(-1), b.reshape(-1))

            distance_func = custom_cosine
        else:
            raise ValueError(f'`how` parameter must be in [euclidean, mae, rmse]')

        real_corr = compute_associations(self.real, nominal_columns=self.categorical_columns, theil_u=True)
        fake_corr = compute_associations(self.fake, nominal_columns=self.categorical_columns, theil_u=True)

        return distance_func(
            real_corr.values,
            fake_corr.values
        )

    def fit_estimators(self):

        for i, est in enumerate(self.r_estimators):
            if self.verbose:
                pass
            est.fit(self.real_x_train, self.real_y_train)
            est.fit(self.fake_x_train, self.fake_y_train)

    def score_estimators(self):
        rows = []
        for r_classifier, f_classifier, estimator_name in zip(self.r_estimators, self.f_estimators, self.estimator_names):
            for dataset, target, dataset_name in zip(
                    [self.real_x_test, self.fake_x_test],
                    [self.real_y_test, self.fake_y_test],
                    ['real', 'fake']
            ):
                predictions_classifier_real = r_classifier.predict(dataset)
                predictions_classifier_fake = f_classifier.predict(dataset)
                f1_r = f1_score(target, predictions_classifier_real, average="micro")
                f1_f = f1_score(target, predictions_classifier_fake, average="micro")
                jac_sim = jaccard_score(predictions_classifier_real, predictions_classifier_fake, average='micro')
                row = {'index': f'{estimator_name}_{dataset_name}_testset', 'f1_real': f1_r, 'f1_fake': f1_f,
                           'jaccard_similarity': jac_sim}
                rows.append(row)
        results = pd.DataFrame(rows).set_index('index')
        return results
