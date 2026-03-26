import os
import sys

sys.path.append(os.path.abspath('../src'))


def load_data():
    import pandas as pd
    return pd.read_csv('../datasets/housing/housing.csv')


def split_data(df):
    import numpy as np
    from sklearn.model_selection import train_test_split

    df['income_cat'] = np.ceil(df['median_income']/1.5)
    df['income_cat'] = df['income_cat'].where(df['income_cat'] < 5, 5.0)

    X = df.drop('median_house_value', axis=1).copy()
    y = df['median_house_value'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=df['income_cat'])

    X_train.drop(columns=['income_cat'])
    X_test.drop(columns=['income_cat'])

    return X_train, X_test, y_train, y_test


def build_pipeline(X_train):
    from utils import CombinedAttributesAdder
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    num_attribs = X_train.select_dtypes(include='number').columns
    cat_attribs = X_train.select_dtypes(include='object').columns

    cat_pipeline = Pipeline([
        ('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
    ])

    num_pipeline = Pipeline([
        ('attrib_add', CombinedAttributesAdder()),
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocess = ColumnTransformer([
        ('cat', cat_pipeline, cat_attribs),
        ('num', num_pipeline, num_attribs),
    ], verbose_feature_names_out=False, )

    return preprocess
