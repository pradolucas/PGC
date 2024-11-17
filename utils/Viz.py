from abc import ABC, abstractmethod

import pandas as pd

from utils.feature import Feature


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

    Parameters:
    -----------
    column_data : pd.DataFrame | pd.Series
        The data to be visualized. This can be either a DataFrame or a Series.
    feature_w_column_data : bool, default=False
        If True, the feature is computed based on the column data; otherwise, it is computed without it.

    Methods:
    --------
    __init__(column_data: pd.DataFrame | pd.Series, feature_w_column_data=False) -> None
        Initializes the visualization object, checks the data type, and computes the feature if required.
    _data_type_check(**kwargs) -> None
        Abstract method to check the data type of the provided data (to be implemented by subclasses).
    get_params() -> dict
        Abstract method to return the parameters of the visualization (to be implemented by subclasses).
    _compute_feature(**kwargs) -> Any
        Abstract method to compute the feature of the data (to be implemented by subclasses).
    plt(**kwargs) -> None
        Abstract method to plot the visualization (to be implemented by subclasses).
    """

    def __init__(
        self, column_data: pd.DataFrame | pd.Series, feature_w_column_data: bool = False
    ) -> None:
        """
        Initializes the visualization object, checks the data type, and computes the feature if required.

        Parameters:
        -----------
        column_data : pd.DataFrame | pd.Series
            The data to be visualized. This can be either a DataFrame or a Series.
        feature_w_column_data : bool, default=False
            If True, the feature is computed based on the column data; otherwise, it is computed without it.

        Raises:
        -------
        TypeError
            If column_data is neither a DataFrame nor a Series.
        """

        if isinstance(column_data, pd.Series):
            # Convert Series to DataFrame
            column_data = column_data.to_frame()
        elif not isinstance(column_data, pd.DataFrame):
            # Raise an error if the input is not a DataFrame or Series
            raise TypeError("Expected column_data to be a pandas DataFrame or Series")

        self.data_type = tuple(
            Feature.get_data_type(data) for _, data in column_data.items()
        )
        self.column_name = tuple(column_name for column_name in column_data)
        if feature_w_column_data:
            self.feature = self._compute_feature(column_data)
        else:
            self.feature = self._compute_feature()

    @abstractmethod
    def _data_type_check(self, data: pd.DataFrame | pd.Series):
        """
        Abstract method to check the data type of the column data.
        Subclasses should implement this method to validate the data types.

        Parameters:
        -----------
        data : pd.DataFrame or pd.Series
            The data to check the type of.
        """


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
    def _compute_feature(self, **kwargs):
        """
        Abstract method to compute a feature of the data.
        This must be implemented by subclasses.

        Returns:
        --------
        Any
            The computed feature based on the data.
        """

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
