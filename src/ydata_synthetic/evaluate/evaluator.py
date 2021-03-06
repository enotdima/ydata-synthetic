import pandas as pd
import numpy as np
import sklearn

from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


def mape(y_true: np.ndarray, y_pred: np.ndarray):
    """
    Calculate MAPE metric.

    :param y_true: array with true values
    :param y_pred: array with predicted values
    :return: MAPE metric for given arrays
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true))


class EvaluateData:
    """
    Contains methods for evaluation of real vs fake data.
    """
    def __init__(
            self,
            real_data: pd.DataFrame,
            fake_data: pd.DataFrame,
            n_samples: int,
            seed=59
    ):
        self.n_samples = n_samples
        self.random_seed = seed
        if n_samples:
            self.real_data = real_data.sample(n_samples)
            self.fake_data = fake_data.sample(n_samples)
        else:
            self.real_data = real_data.copy()
            self.fake_data = fake_data.copy()

    def pca_correlation(self, n_components=5):
        """"
        """
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
        # Print the results
        print(f'Top {n_components} PCA components:')
        print(results.to_string())
        # Count metric
        pca_mape = 1 - mape(pca_real.explained_variance_, pca_fake.explained_variance_)
        return pca_mape

    @staticmethod
    def fit_estimators(
            real_estimators,
            fake_estimators,
            real_features_train,
            real_target_train,
            fake_features_train,
            fake_target_train
    ):
        """
        Fit ML estimators based on true and fake data.

        :param real_estimators: algorithms for real data
        :param fake_estimators: algorithms for fake data
        :param real_features_train: real train data
        :param real_target_train: real data targets
        :param fake_features_train: fake train data
        :param fake_target_train: fake data targets
        :return: trained real and fake estimators
        """
        for i, est in enumerate(real_estimators):
            real_estimators[i] = est.fit(real_features_train, real_target_train)
        for i, est in enumerate(fake_estimators):
            fake_estimators[i] = est.fit(fake_features_train, fake_target_train)
        return real_estimators, fake_estimators

    @staticmethod
    def score_estimators(
            real_estimators,
            fake_estimators,
            real_features_test,
            fake_features_test,
            real_target_test,
            fake_target_test
    ):
        """
        Score the ML algorithms performance with F1 score metrics.

        :param real_estimators:
        :param fake_estimators:
        :param real_features_test:
        :param fake_features_test:
        :param real_target_test:
        :param fake_target_test:
        :return: dataframe with F1 scores for fake and real data and MAPE score between them
        """
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
        """
        Create and train ML estimators to evaluate GAN performance.

        :param target_column: the target column name
        :return: MAPE results of ML evaluation
        """
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
