import pandas as pd
import dask
import dask.dataframe as dd
import dask_ml

from dask_ml.preprocessing import Categorizer, OneHotEncoder, StandardScaler
from dask_ml.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from pmlb import fetch_data

def transformations():
    dask_data = fetch_data('adult')
    numerical_features = ['age', 'fnlwgt',
                          'capital-gain', 'capital-loss',
                          'hours-per-week']
    numerical_transformer = Pipeline(steps=[
        ('onehot', StandardScaler())])

    categorical_features = ['workclass','education', 'marital-status',
                            'occupation', 'relationship',
                            'race', 'sex']
    categorical_transformer = Pipeline(steps=[
        ('cat', Categorizer(columns=categorical_features)),
        ('onehot', OneHotEncoder())
    ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
            ]
    )
    processed_data = preprocessor.fit_transform(dask_data)
    processed_data['target'] = dask_data['target']

    return processed_data, preprocessor
