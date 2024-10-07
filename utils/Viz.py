from abc import abstractmethod, ABCMeta
from utils.feature import feature


class Viz():
    __metaclass__ = ABCMeta
    
    def __init__(self, column_data):
        self.data_type = feature.get_data_type(column_data) # Turn datatypes into class
        self.column_name = column_data.name
        self.feature   = self._compute_feature()
        # self.columns   = columns
    
    def __init__(self, x, y):
        self.data_type = feature.get_data_type(x), feature.get_data_type(y)  # Turn datatypes into class
        self.column_name = x.name, y.name
        self.feature   = self._compute_feature()
        # self.columns   = columns
        
    # @classmethod
    # def two_D(cls):
    #     return cls(randint(66, 100))
          
    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def _compute_feature(self):
        pass

    @abstractmethod
    def plt(self, **kwargs):
        pass
    
