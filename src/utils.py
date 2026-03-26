"""
Módulo de utilitários para o projeto California Housing.
Contém transformadores customizados para o pipeline de dados e funções auxiliares para avaliação de métricas.
"""
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Transformador customizado do scikit-learn que cria novas features combinadas
    específicas para o dataset California Housing.

    Attributes:
        add_bedrooms_per_room (bool): Indica se a feature 'bedrooms_per_room' deve ser adicionada.
                                      Útil para testar se essa feature melhora o modelo no GridSearchCV.
    """

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        X['rooms_per_household'] = X['total_rooms']/X['households']
        X['population_per_household'] = X['population']/X['households']

        if self.add_bedrooms_per_room:
            X['bedrooms_per_room'] = X['total_bedrooms']/X['total_rooms']

        return X

    def get_feature_names_out(self, input_features=None):
        return list(input_features) + [
            'rooms_per_household',
            'population_per_household',
            'bedrooms_per_room'
        ]


# train.py
def display_results(y_test, y_pred):
    """
        Calcula, imprime na tela e retorna as principais métricas de regressão.

        Args:
            y_test (array-like): Os valores reais.
            y_pred (array-like): Os valores preditos.

        Returns:
            dict: Um dicionário contendo as métricas calculadas ('mse', 'rmse', 'mae', 'r2').
    """

    mse = mean_squared_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f'MSE: {round(mse,4)}')
    print(f'RMSE: {round(rmse,4)}')
    print(f'MAE: {round(mae,4)}')
    print(f'R2: {round(r2,4)}')

    metrics = {'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2}

    return metrics
