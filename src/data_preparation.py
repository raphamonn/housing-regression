import os
import sys

sys.path.append(os.path.abspath('../src'))


def load_data():
    import pandas as pd
    return pd.read_csv('../datasets/housing/housing.csv')


def split_data(df, remove_outliers=False):
    import numpy as np
    from sklearn.model_selection import train_test_split

    df['income_cat'] = np.ceil(df['median_income']/1.5)
    df['income_cat'] = df['income_cat'].where(df['income_cat'] < 5, 5.0)

    print('Df shape:', df.shape)

    if remove_outliers:
        df = df[~(df['median_house_value'] == 500001) &
                ~(df['housing_median_age'] == 52)].copy()
        print('Shape após remoção de outliers', df.shape)

    X = df.drop('median_house_value', axis=1).copy()
    y = df['median_house_value'].copy()

    X_train, X_test, y_train, y_test = train_test_split(X,
                                                        y,
                                                        test_size=0.2,
                                                        random_state=42,
                                                        stratify=(
                                                            df['income_cat']))

    X_train.drop(columns=['income_cat'], inplace=True)
    X_test.drop(columns=['income_cat'], inplace=True)

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    from utils import CombinedAttributesAdder
    from sklearn.impute import SimpleImputer  # type: ignore
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV

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

    train_pipeline = Pipeline([
        ('preprocess', preprocess),
        ('model', RandomForestRegressor(random_state=42))
    ])

    param_grid = {'model__n_estimators': [110, 170, 200, 230],
                  'model__max_features': [10, 12, 'sqrt', 'log2'],
                  }

    grid_search = GridSearchCV(
        estimator=train_pipeline,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=2,
        error_score='raise'
    )

    grid_search.fit(X_train, y_train)

    return grid_search
