from math import log, e
import numpy as np
import pandas as pd


class Feature():

    @staticmethod
    def get_count(data):
        """
        equivalent to return len(data), data.value_counts()
        """
        n_labels = len(data)
        value, counts = np.unique(data, return_counts=True)
        return n_labels, (value, counts)

    @staticmethod
    def entropy_numpy(n_labels, freq, base=2):
        # def entropy_numpy(probs, base=2):
        """Computes entropy of label distribution with numpy."""

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
            if pd.api.types.is_integer_dtype(series) and unique_values / len(series) < 0.1:
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