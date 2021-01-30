import pandas as pd
import numpy as np
import sklearn

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from scipy import stats


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray):
    return np.mean(np.abs(np.subtract(y_true, y_pred)))


def euclidean_distance(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(np.sum(np.power(np.subtract(y_true, y_pred), 2)))


def mape(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


def rmse(y_true: np.ndarray, y_pred: np.ndarray):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def cosine_similarity(y_true: np.ndarray, y_pred: np.ndarray):
    y_true, y_pred = y_true.reshape(-1), y_pred.reshape(-1)
    return np.sum(y_true * y_pred) / (np.sqrt(np.sum(y_true ** 2)) * np.sqrt(np.sum(y_pred ** 2)))


class EvaluateData:
    """
    Contains methods for evaliation of real vs fake data.
    """

    def __init__(
            self,
            real_data: pd.DataFrame,
            fake_data: pd.DataFrame,
            n_samples: int,
            metric='pearson',
            seed=59
    ):
        self.n_samples = n_samples
        self.comparison_metric = getattr(stats, metric)
        self.random_seed = seed
        if n_samples:
            self.real_data = real_data.sample(n_samples)
            self.fake_data = fake_data.sample(n_samples)
        else:
            self.real_data = real_data.copy()
            self.fake_data = fake_data.copy()

    def pca_correlation(self, n_components=5):
        # Initialize PCA
        pca_real = PCA(n_components=n_components)
        pca_fake = PCA(n_components=n_components)
        # Fit PCA on real and fake data
        pca_real.fit(self.real_data.to_numpy())
        pca_fake.fit(self.fake_data.to_numpy())
        results = pd.DataFrame(
            {
                'real_data_pca': pca_real.explained_variance_,
                'fake_data_pca': pca_fake.explained_variance_
            }
        )

        print(f'Top {n_components} PCA components:')
        print(results.to_string())

        pca_mape = 1 - mape(pca_real.explained_variance_, pca_fake.explained_variance_)
        return pca_mape

    def fit_estimators(self, real_estimators, fake_estimators, real_features_train, real_target_train,
                       fake_features_train, fake_target_train):
        """
        Fit given estimators.
        """
        for i, est in enumerate(real_estimators):
            real_estimators[i] = est.fit(real_features_train, real_target_train)
        for i, est in enumerate(fake_estimators):
            fake_estimators[i] = est.fit(fake_features_train, fake_target_train)
        return real_estimators, fake_estimators

    def score_estimators(
            self,
            real_estimators,
            fake_estimators,
            real_features_test,
            fake_features_test,
            real_target_test,
            fake_target_test
    ):
        rows = []
        for real_est, fake_est in zip(real_estimators, fake_estimators):
            for dataset, target, dataset_name in zip(
                    [real_features_test, fake_features_test],
                    [real_target_test, fake_target_test],
                    ['real', 'fake']
            ):
                predictions_classifier_real = real_est.predict(dataset)
                predictions_classifier_fake = fake_est.predict(dataset)
                row = {'index': f'{real_est.__class__.__name__}_{dataset_name}_test',
                       'real_data_f1': f1_score(target, predictions_classifier_real, average="micro"),
                       'fake_data_f1': f1_score(target, predictions_classifier_fake, average="micro"),
                       }
                rows.append(row)
        results = pd.DataFrame(rows).set_index('index')
        metric = 1 - mape(results['real_data_f1'], results['fake_data_f1'])
        return results, metric

    def set_ml_estimation(self, target_column="target"):
        real_features = self.real_data.drop([target_column], axis=1)
        real_target = self.real_data[target_column]
        fake_features = self.fake_data.drop([target_column], axis=1)
        fake_target = self.fake_data[target_column]
        np.random.seed(self.random_seed)
        real_features_train, real_features_test, real_target_train, real_target_test = train_test_split(
            real_features, real_target
        )
        fake_features_train, fake_features_test, fake_target_train, fake_target_test = train_test_split(
            fake_features, fake_target
        )
        real_estimators = [
            sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=self.random_seed),
            DecisionTreeClassifier(random_state=self.random_seed),
            RandomForestClassifier(n_estimators=12, random_state=self.random_seed),
        ]

        fake_estimators = [
            sklearn.linear_model.LogisticRegression(max_iter=1000, random_state=self.random_seed),
            DecisionTreeClassifier(random_state=self.random_seed),
            RandomForestClassifier(n_estimators=12, random_state=self.random_seed),
        ]

        real_estimators, fake_estimators = self.fit_estimators(
            real_estimators,
            fake_estimators,
            real_features_train,
            real_target_train,
            fake_features_train,
            fake_target_train
        )
        results, score = self.score_estimators(
            real_estimators,
            fake_estimators,
            real_features_train,
            fake_features_train,
            real_target_train,
            fake_target_train
        )
        return results, score
