
import joblib
from data_preparation import load_data, split_data
import matplotlib.pyplot as plt
from utils import display_results

model = joblib.load('../models/final_model.pkl')
df = load_data()
X_train, X_test, y_train, y_test = split_data(df)
y_pred = model.predict(X_test)

# %%
results = display_results(y_test, y_pred)
# %%
plt.figure(figsize=(10, 6))

plt.scatter(x=y_test, y=y_pred, alpha=0.5,
            c='blue', edgecolors='black'
            )

# (R² = 1)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         color='red', linestyle='--', lw=2)

plt.title('Preço Real vs Preço predito pelo Modelo', fontsize=14)
plt.xlabel('', fontsize=12)
plt.legend(['Dados Reais', 'R2'])
plt.ylabel('Preço Real', fontsize=12)
plt.savefig('../imgs/result.png')

plt.show()
