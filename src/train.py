from data_preparation import load_data, split_data, build_pipeline
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import joblib

df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
preprocess = build_pipeline(X_train)

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
final_model = grid_search.best_estimator_
final_model.fit(X_train, y_train)

joblib.dump(final_model, '../models/final_model.pkl')
