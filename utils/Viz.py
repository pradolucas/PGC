from abc import abstractmethod, ABCMeta
from utils.feature import feature


from abc import ABC, abstractmethod, ABCMeta
import matplotlib.pyplot as plt

class Viz():
    __metaclass__ = ABCMeta
    
    # def __init__(self, column_data):
    #     self.data_type = feature.get_data_type(column_data) # Turn datatypes into class
    #     self.column_name = column_data.name
    #     # print(column_data)
    #     self.feature   = self._compute_feature()
        # self.columns   = columns
    
    # def __init__(self, x, y):
    #     self.data_type = feature.get_data_type(x), feature.get_data_type(y)  # Turn datatypes into class
    #     self.column_name = x.name, y.name
    #     self.feature   = self._compute_feature()
    #     # self.columns   = columns
    
    def __init__(self, column_data: pd.DataFrame | pd.Series):
        
        try:
            if isinstance(column_data, pd.Series):
                column_data = column_data.to_frame()
            elif not isinstance(column_data, pd.DataFrame):
                raise ValueError("Not a valid column_data format")
        except:
            pass
        
        self.data_type = tuple(feature.get_data_type(data) for _, data in column_data.items())
        self.column_name = tuple(column_name for column_name in column_data)
        self.feature   = self._compute_feature()
          
    # @classmethod
    # def ndim(cls, *args):
    #     ## instantiate obj as Viz.ndim(x, y ...)
    #     return cls()
          
    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def _compute_feature(self):
        pass

    @abstractmethod
    def plt(self, **kwargs):
        pass
    
