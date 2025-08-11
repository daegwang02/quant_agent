# foundations/base.py

import abc
import numpy as np
import pandas as pd


class ExpressionOps(abc.ABC):
    def __call__(self, *args, **kwargs):
        return self.load(*args, **kwargs)

    @abc.abstractmethod
    def load(self, instrument, start_index, end_index, *args, **kwargs):
        pass

    def __str__(self):
        return self.__class__.__name__

    @abc.abstractmethod
    def get_longest_back_rolling(self):
        pass

    @abc.abstractmethod
    def get_extended_window_size(self):
        pass


class Expression(ExpressionOps):
    def __init__(self, feature_list, freq):
        self.feature_list = feature_list
        self.freq = freq

    def __str__(self):
        return "{}({})".format(self.__class__.__name__, self.feature_list)


class Feature(Expression):
    def __init__(self, feature, freq="day"):
        super(Feature, self).__init__(feature, freq)
        self.feature = feature

    def load(self, instrument, start_index, end_index, *args, **kwargs):
        # NOTE: This is a simplified version for demonstration
        # In a real-world scenario, this would load data from a database
        return pd.Series(np.random.rand(end_index - start_index), 
                         index=pd.MultiIndex.from_product([[pd.Timestamp('2021-01-01') + pd.DateOffset(days=i) for i in range(start_index, end_index)], [instrument]], names=['date', 'ticker']))

    def get_longest_back_rolling(self):
        return 0

    def get_extended_window_size(self):
        return 0, 0


class PFeature(Feature):
    pass
