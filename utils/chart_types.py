import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, skew
from utils.feature import Feature
from utils.viz import Viz


class Histogram(Viz):

    def __init__(self, column_data: pd.Series):
        self.accept_data_type = ["Continuous"]
        self._data_type_check(column_data)
        self.data_len = len(column_data)
        self.frequency, self.bins = Histogram._calculate_hist_params(column_data)
        super().__init__(column_data)

    def _data_type_check(self, column_data):
        data_type = Feature.get_data_type(column_data)
        if data_type not in self.accept_data_type:
            raise TypeError(
                f"Column '{column_data.name}' type {data_type} not one of the accepted data types: {self.accept_data_type}"
            )

    # TODO replace frequency, bin to params in init
    @staticmethod
    def _calculate_hist_params(column_data: pd.Series) -> dict:
        frequency, bins = np.histogram(
            column_data,
            bins=10,
            range=[
                (column_data.min() if column_data.min() < 0 else 0),
                column_data.max(),
            ],
            density=False,
        )
        # return {"frequency": frequency, "bins": bins}
        return frequency, bins

    def _compute_feature(self):
        """Calculates entropy"""
        # TODO timeit
        # scipy_entr = feature.entropy_scipy(self.data_len, self.frequency)
        numpy_entr = Feature.entropy_numpy(self.data_len, self.frequency)

        # print(scipy_entr, numpy_entr)
        return numpy_entr

    def get_params(self):
        # return self.feature, self.data_len, self.data_freq
        return {"feature": self.feature, "params": (self.frequency, self.bins)}

    def plt(self, **kwargs):
        if "axs" in kwargs:
            ax = kwargs["axs"]
        else:
            # TODO Printar linha y com a distribuição
            _, ax = plt.subplots()
        title_idx = f"({kwargs['title_idx']})" if "title_idx" in kwargs else ""
        ax.bar(
            self.bins[:-1],
            self.frequency,
            width=np.diff(self.bins),
            edgecolor="black",
            align="edge",
        )
        ax.set_title(f"{self.column_name[0]} {title_idx}")


class BoxPlot(Viz):

    def __init__(self, column_data):
        self.accept_data_type = ["Continuous", "Discrete"]
        self._data_type_check(column_data)

        self.data_len = len(column_data)
        self.params = BoxPlot._calculate_boxplot_params(column_data)
        super().__init__(column_data, feature_w_column_data=True)

    def _data_type_check(self, column_data):
        data_type = Feature.get_data_type(column_data)
        if data_type not in self.accept_data_type:
            raise TypeError(
                f"Column '{column_data.name}' type {data_type} not one of the accepted data types: {self.accept_data_type}"
            )

    @staticmethod
    def _calculate_boxplot_params(series: pd.Series) -> dict:
        """
        Calculate the boxplot parameters (min, Q1, median, Q3, max, and outliers) from a pandas Series.
        """
        q1 = series.quantile(0.25)
        median = series.median()
        q3 = series.quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        whisker_low = q1 - 1.5 * iqr
        whisker_high = q3 + 1.5 * iqr

        # Identify outliers (fliers)
        outliers = series[(series < whisker_low) | (series > whisker_high)].values

        return {
            "whislo": whisker_low,  # Bottom whisker position
            "q1": q1,  # First quartile (25th percentile)
            "med": median,  # Median (50th percentile)
            "q3": q3,  # Third quartile (75th percentile)
            "whishi": whisker_high,  # Top whisker position
            "fliers": outliers,  # Outliers
        }

    @staticmethod
    def skewness_test(series: pd.Series):
        # TODO turn in static method
        # TODO change docstring
        """
        Give the skewness of a Pandas Serie

        Parameters:
            series (pd.Series): The data series to test.
        Returns:
            float: The skewness of the series.
        """
        # Calculate skewness
        skewness_value = skew(series)[0]

        return abs(skewness_value)

    def outlier_percentage(self):
        n_outliers = len(self.params["fliers"])
        return (n_outliers / self.data_len) * 100

    def _compute_feature(self, series):
        """Calculates skewness"""
        skewness_value = BoxPlot.skewness_test(series)
        outlier_percentage = self.outlier_percentage()
        return (skewness_value + 1) * (outlier_percentage + 0.5)

    def get_params(self):
        return {"feature": self.feature, **self.params}

    def plt(self, **kwargs):
        if "axs" in kwargs:
            ax = kwargs["axs"]
        else:
            # plt.boxplot(column_data, patch_artist=True)  # fill with random  color
            _, ax = plt.subplots()
        ax.bxp(
            [self.params],
            showfliers=True,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue"),
        )
        ax.set_xticklabels([*self.column_name])  # Rotaciona o rótulo do eixo x
        if "title_idx" in kwargs:
            ax.set_title(f"({kwargs['title_idx']})")


class Scatter(Viz):

    def __init__(self, data: pd.DataFrame):
        self.accept_data_type = ["Continuous"]
        self._data_type_check(data)
        self.x, self.y = data.T.values
        super().__init__(data)

    def _data_type_check(self, data):
        for _, column_data in data.items():
            data_type = Feature.get_data_type(column_data)
            if data_type not in self.accept_data_type:
                raise TypeError(
                    f"Column '{column_data.name}' type {data_type} not one of the accepted data types: {self.accept_data_type}"
                )
    
    def _compute_feature(self):
        """Calculates corr"""
        # TODO timeit
        corr = pearsonr(self.x, self.y).statistic
        return corr

    def get_params(self):
        return {"feature": self.feature, "params": None}

    def plt(self, **kwargs):
        if "axs" in kwargs:
            ax = kwargs["axs"]
        else:
            # Printar linha y com a distribuição
            _, ax = plt.subplots()
        title_idx = f"({kwargs['title_idx']})" if "title_idx" in kwargs else ""
        ax.scatter(self.x, self.y)
        ax.text(0.05, 0.8, "r={:.2f}".format(self.feature), transform=ax.transAxes)
        ax.set_title(f"{self.column_name[0]} x {self.column_name[1]} {title_idx}")
