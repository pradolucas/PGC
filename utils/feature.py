from math import log

import numpy as np
import pandas as pd


class Feature:
    """
    A class providing static methods for analyzing and processing feature data.

    Methods:
    --------
    get_count(data: pd.Series) -> tuple
        Computes the number of labels and their frequencies in the provided data.

    entropy_numpy(n_labels: int, freq: np.ndarray, base: int = 2) -> float
        Computes the entropy of the label distribution using numpy.

    get_data_type(series: pd.Series) -> str
        Determines the data type of the provided pandas Series (e.g., "Binary", "Discrete", "Continuous", "Text").
    """

    @staticmethod
    def get_count(data: pd.Series) -> tuple:
        """
        Computes the number of labels and their frequencies in the provided data.

        Parameters:
        -----------
        data : pd.Series
            The input data to compute the count and value frequencies.

        Returns:
        --------
        tuple
            A tuple containing the number of labels (int) and a tuple of unique values and their counts (np.ndarray).
        """
        n_labels = len(data)
        value, counts = np.unique(data, return_counts=True)
        return n_labels, (value, counts)

    @staticmethod
    def entropy_numpy(n_labels: int, freq: np.ndarray, base: int = 2) -> float:
        """
        Computes the entropy of the label distribution using numpy.

        Parameters:
        -----------
        n_labels : int
            The total number of labels in the data.
        freq : np.ndarray
            The frequency of each unique label.
        base : int, optional
            The logarithmic base for the entropy calculation (default is 2).

        Returns:
        --------
        float
            The computed entropy of the label distribution.
        """

        if n_labels <= 1:
            return 0
        probs = freq / n_labels
        probs = probs[probs > 0]

        n_classes = np.count_nonzero(probs)
        if n_classes <= 1:
            return 0

        # Compute entropy
        # base = e if base is None else base
        ent = 0.0
        for prob in probs:
            ent -= prob * log(prob, base)
        return ent

    @staticmethod
    def get_data_type(series: pd.Series) -> str:
        """
        Determines the data type of the provided pandas Series.

        Parameters:
        -----------
        series : pd.Series
            The input pandas Series whose data type needs to be determined.

        Returns:
        --------
        str
            The data type of the Series: one of "Empty", "Datetime", "Binary", "Discrete", "Continuous", "Categorical", "Text", or "Unknown".
        """
        # Check if the series is empty
        if series.empty:
            return "Empty"

        # Check if the series is of datetime type
        if pd.api.types.is_datetime64_any_dtype(series):
            return "Datetime"

        # Check if the series is boolean
        if pd.api.types.is_bool_dtype(series):
            return "Binary"

        unique_values = series.nunique()
        # Check if the series is numeric (int or float)
        if pd.api.types.is_numeric_dtype(series):
            # If there are only 2 unique values, treat as binary
            if unique_values == 2:
                return "Binary"

            # If integer type and unique values are relatively low, consider it discrete
            if (
                pd.api.types.is_integer_dtype(series)
                and unique_values / len(series) < 0.1
            ):
                return "Discrete"

            # Otherwise, treat as continuous
            return "Continuous"

        # If the series is an object (e.g., strings or mixed types)
        if pd.api.types.is_object_dtype(series):
            total_values = len(series)

            # Consider the series categorical if unique values are relatively small
            if unique_values / total_values < 0.05:
                return "Categorical"
            else:
                return "Text"

        return "Unknown"
