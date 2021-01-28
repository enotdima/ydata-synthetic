import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy import stats

from sklearn.metrics import f1_score, mean_squared_error, jaccard_score



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
        pca_real.fit(self.real_data)
        pca_fake.fit(self.fake_data)
        results = pd.DataFrame(
            {
                'real_data_pca': pca_real.explained_variance_,
                'fake_data_pca': pca_fake.explained_variance_
            }
        )

        print(f'Top {n_components} PCA components:')
        print(results.to_string())

        pca_mape = 1 - mean_absolute_percentage_error(pca_real.explained_variance_, pca_fake.explained_variance_)
        return pca_mape

    def fit_estimators(self):
        """
        Fit given estimators.
        """
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
                row = {'index': f'{estimator_name}_{dataset_name}_testset',
                       'f1_real': f1_r, 'f1_fake': f1_f,
                        'jaccard_similarity': jac_sim}
                rows.append(row)
        results = pd.DataFrame(rows).set_index('index')
        return results
