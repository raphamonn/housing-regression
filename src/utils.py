# data_preparation
from sklearn.base import BaseEstimator, TransformerMixin


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
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


# train
