# %%
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
        ('onehot', OneHotEncoder(sparse_output=False))
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


# # %%
# df = load_data()
# X_train, X_test, y_train, y_test = split_data(df)
# preprocess = build_pipeline(X_train)
# # %%
# preprocess.fit(X_train)
# preprocess.get_feature_names_out()
# # %%
# X_test_processed = preprocess.transform(X_test)

# extra_attribs = ['rooms_per_household',
#                  'population_per_household', 'bedrooms_per_room']
# onehot_encode_attribs = list(preprocess.named_transformers_[
#                              'cat']['onehot'].categories_[0])
# cols = onehot_encode_attribs + num_attribs.to_list() + extra_attribs


# X_train_processed = pd.DataFrame(X_train_processed, columns=cols)
# X_test_processed = pd.DataFrame(X_test_processed, columns=cols)
# # %%
# preprocess.get_feature_names_out()
# # %%
