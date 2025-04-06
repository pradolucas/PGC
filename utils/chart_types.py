import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import pearsonr, skew

from utils.feature import Feature
from utils.viz import Viz


# class Charts():
class Histogram(Viz):
    """
    A class for creating histograms for continuous data.

    Parameters:
    -----------
    column_data : pd.Series
        The data to be visualized as a histogram.

    Methods:
    --------
    _data_type_check(column_data: pd.Series) -> None
        Checks if the data type of the provided column is valid for histogram visualization.

    _calculate_hist_params(column_data: pd.Series) -> tuple
        Calculates the frequency and bin parameters for the histogram.

    _compute_feature() -> float
        Computes the entropy of the data distribution.

    get_params() -> dict
        Returns the calculated features and parameters for the histogram.

    plt(**kwargs) -> None
        Plots the histogram.
    """

    def __init__(self, column_data: pd.Series):
        self.accept_data_type = ["Continuous", "Discrete", "Datetime"]
        self._data_type_check(column_data)
        self.data_len = len(column_data)
        self.frequency, self.bins = Histogram._calculate_hist_params(column_data)
        super().__init__(column_data)

    def _data_type_check(self, data: pd.Series) -> None:
        """
        Checks if the data type of the provided column is valid for histogram visualization.

        Parameters:
        -----------
        data : pd.Series
            The data to check the type.

        Raises:
        -------
        TypeError
            If the data type is not valid for histogram visualization.
        """
        data_type = Feature.get_data_type(data)
        if data_type not in self.accept_data_type:
            raise TypeError(
                f"Column '{data.name}' type {data_type} not one of the accepted data types: {self.accept_data_type}"
            )

    @staticmethod
    def _calculate_hist_params(column_data: pd.Series) -> tuple:
        """
        Calculates the frequency and bin parameters for the histogram.

        Parameters:
        -----------
        column_data : pd.Series
            The data to calculate histogram parameters.

        Returns:
        --------
        tuple
            A tuple containing the frequency and bins for the histogram.
        """
        frequency, bins = np.histogram(
            column_data,
            bins=10,
            range=[
                (column_data.min() if column_data.min() < 0 else 0),
                column_data.max(),
            ],
            density=False,
        )
        return frequency, bins

    def _compute_feature(self) -> float:
        """
        Computes the entropy of the data distribution.

        Returns:
        --------
        float
            The calculated entropy of the distribution.
        """
        numpy_entr = Feature.entropy_numpy(self.data_len, self.frequency)  # TODO timeit
        return numpy_entr

    def get_params(self) -> dict:
        """
        Returns the calculated features and parameters for the histogram.

        Returns:
        --------
        dict
            A dictionary containing the feature and histogram parameters.
        """
        return {"feature": self.feature, "params": (self.frequency, self.bins)}

    def plt(self, **kwargs) -> None:
        """
        Plots the histogram.

        Parameters:
        -----------
        kwargs : dict
        """
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

# TODO change doc
class BoxPlot(Viz):
    """
    A class for creating box plots for continuous or discrete data.

    Parameters:
    -----------
    column_data : pd.Series
        The data to be visualized as a box plot.

    Methods:
    --------
    _data_type_check(column_data: pd.Series) -> None
        Checks if the data type of the provided column is valid for box plot visualization.

    _calculate_boxplot_params(series: pd.Series) -> dict
        Calculates the parameters for the box plot (min, Q1, median, Q3, max, outliers).
    
    skewness_test(series: pd.Series) -> float
        Calculates the skewness of the data.

    outlier_percentage() -> float
        Computes the percentage of outliers in the data.

    _compute_feature(series: pd.Series) -> float
        Computes a feature based on skewness and outlier percentage.

    get_params() -> dict
        Returns the calculated features and parameters for the box plot.

    plt(**kwargs) -> None
        Plots the box plot.
    """

    def __init__(self, column_data):
        self.accept_data_type = ["Continuous", "Discrete"]
        self._data_type_check(column_data)

        self.data_len = len(column_data)
        self.params = BoxPlot._calculate_boxplot_params(column_data)
        super().__init__(column_data, feature_w_column_data=True)

    def _data_type_check(self, data: pd.Series) -> None:
        """
        Checks if the data type of the provided column is valid for box plot visualization.

        Parameters:
        -----------
        data : pd.Series
            The data to check the type.

        Raises:
        -------
        TypeError
            If the data type is not valid for box plot visualization.
        """
        data_type = Feature.get_data_type(data)
        if data_type not in self.accept_data_type:
            raise TypeError(
                f"Column '{data.name}' type {data_type} not one of the accepted data types: {self.accept_data_type}"
            )

    @staticmethod
    def _calculate_boxplot_params(series: pd.Series) -> dict:
        """
        Calculates the parameters for the box plot (min, Q1, median, Q3, max, outliers).

        Parameters:
        -----------
        series : pd.Series
            The data to calculate box plot parameters.

        Returns:
        --------
        dict
            A dictionary containing the parameters of the box plot.
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
    def skewness(series: pd.Series) -> float:
        """
        Calculates the skewness of the data.

        Parameters:
        -----------
        series : pd.Series
            The data series to calculate skewness for.

        Returns:
        --------
        float
            The absolute skewness value.
        """
        # Calculate skewness
        skewness_value = skew(series.dropna())[0]

        return abs(skewness_value)

    def outlier_percentage(self) -> float:
        """
        Computes the percentage of outliers in the data.

        Returns:
        --------
        float
            The percentage of outliers in the data.
        """
        n_outliers = len(self.params["fliers"])
        return (n_outliers / self.data_len) * 100

    def _compute_feature(self, series: pd.Series) -> float:
        """
        Computes a feature based on skewness and outlier percentage.

        Parameters:
        -----------
        series : pd.Series
            The data series to compute the feature for.

        Returns:
        --------
        float
            The computed feature value.
        """
        skewness_value = BoxPlot.skewness(series)
        outlier_percentage = self.outlier_percentage()
        return (skewness_value + 1) * (outlier_percentage + 0.5)

    def get_params(self) -> dict:
        """
        Returns the calculated features and parameters for the box plot.

        Returns:
        --------
        dict
            A dictionary containing the feature and box plot parameters.
        """
        return {"feature": self.feature, **self.params}

    def plt(self, **kwargs) -> None:
        """
        Plots the box plot.

        Parameters:
        -----------
        kwargs : dict
            Additional keyword arguments for plotting, including 'axs' for axes and 'title_idx' for the title.
        """
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
    """
    A class for creating scatter plots for continuous data.

    Parameters:
    -----------
    data : pd.DataFrame
        The data to be visualized as a scatter plot.

    Methods:
    --------
    _data_type_check(data: pd.DataFrame) -> None
        Checks if the data type of the provided columns is valid for scatter plot visualization.

    _compute_feature() -> float
        Computes the Pearson correlation between the two columns of data.

    get_params() -> dict
        Returns the calculated features and parameters for the scatter plot.

    plt(**kwargs) -> None
        Plots the scatter plot.
    """

    def __init__(self, data: pd.DataFrame):
        self.accept_data_type = [("Continuous", "Continuous"), ("Discrete", "Continuous"), ("Continuous","Datatime")]
        self._data_type_check(data)
        self.x, self.y = data.T.values
        super().__init__(data)

    def _data_type_check(self, data: pd.DataFrame) -> None:
        """
        Checks if the data type of the provided columns is valid for scatter plot visualization.

        Parameters:
        -----------
        data : pd.DataFrame
            The data to check the type.

        Raises:
        -------
        TypeError
            If the data type is not valid for scatter plot visualization.
        """
        data_type = tuple(data.apply(Feature.get_data_type))
        if data_type not in self.accept_data_type:
            raise TypeError(
                    f"Columns '{list(data.columns.values)}' type {data_type} not one of the accepted data types: {self.accept_data_type}"
                )

    def _compute_feature(self) -> float:
        """
        Computes the Pearson correlation between the two columns of data.

        Returns:
        --------
        float
            The Pearson correlation coefficient.
        """
        df = pd.DataFrame({'x': self.x, 'y': self.y}).dropna()

        # corr = pearsonr(df["x"], df["y"]).statistic
        # corr = df.corr(method='pearson').iloc[0, 1]
        corr = np.corrcoef(df["x"], df["y"])[0, 1]
        return corr

    def get_params(self) -> dict:
        """
        Returns the calculated features and parameters for the scatter plot.

        Returns:
        --------
        dict
            A dictionary containing the feature and parameters for the scatter plot.
        """
        return {"feature": self.feature, "params": None}

    def plt(self, **kwargs) -> None:
        """
        Plots the scatter plot.

        Parameters:
        -----------
        kwargs : dict
            Additional keyword arguments for plotting, including 'axs' for axes and 'title_idx' for the title.
        """
        if "axs" in kwargs:
            ax = kwargs["axs"]
        else:
            # Printar linha y com a distribuição
            _, ax = plt.subplots()
        title_idx = f"({kwargs['title_idx']})" if "title_idx" in kwargs else ""
        ax.scatter(self.x, self.y)
        ax.text(0.05, 0.8, "r={:.2f}".format(self.feature), transform=ax.transAxes)
        ax.set_title(f"{self.column_name[0]} x {self.column_name[1]} {title_idx}")
