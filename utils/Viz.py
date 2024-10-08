from abc import abstractmethod, ABC, ABCMeta
from utils.feature import feature
import pandas as pd


class Viz(ABC):
    """
    Abstract base class for creating different types of visualizations.
    
    Attributes:
    -----------
    data_type : tuple
        A tuple representing the data type of the columns in the DataFrame or Series.
    column_name : tuple
        A tuple containing the names of the columns.
    feature : Any
        The computed feature for the visualization (defined in child classes).
    """
    
    def __init__(self, column_data: pd.DataFrame | pd.Series):
        
        if isinstance(column_data, pd.Series):
            # Convert Series to DataFrame
            column_data = column_data.to_frame()
        elif not isinstance(column_data, pd.DataFrame):
            # Raise an error if the input is not a DataFrame or Series
            raise TypeError("Expected column_data to be a pandas DataFrame or Series")
        
        self.data_type   = tuple(feature.get_data_type(data) for _, data in column_data.items())
        self.column_name = tuple(column_name for column_name in column_data)
        self.feature     = self._compute_feature()
          
    @abstractmethod
    def get_params(self):
        """
        Abstract method to return the parameters of the visualization.
        This must be implemented by subclasses.
        
        Returns:
        --------
        dict
            A dictionary of parameters related to the visualization.
        """

    @abstractmethod
    def _compute_feature(self):
        """
        Abstract method to compute a feature of the data.
        This must be implemented by subclasses.
        
        Returns:
        --------
        Any
            The computed feature based on the data.
        """
        pass

    @abstractmethod
    def plt(self, **kwargs):
        """
        Abstract method to plot the visualization.
        This must be implemented by subclasses.

        Parameters:
        -----------
        **kwargs : optional
            Additional keyword arguments for customizing the plot.
        """
        pass