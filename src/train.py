# %%
from data_preparation import load_data, split_data, train_model
import joblib

df = load_data()
X_train, X_test, y_train, y_test = split_data(df, True)

# %%
model = train_model(X_train, y_train)
joblib.dump(model.best_estimator_, '../models/final_model.pkl')

# %%
