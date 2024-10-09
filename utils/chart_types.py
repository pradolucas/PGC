from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils.feature import feature
from utils.Viz import Viz
from scipy.stats import pearsonr


class Histogram(Viz):

    def __init__(self, column_data: pd.Series):
        #
        # self.data_len, self.data_freq = feature.get_count(column_data)
        self.data_len = len(column_data)
        self.frequency, self.bins = np.histogram(column_data, bins=10, range=[(column_data.min() if column_data.min() < 0 else 0 ), column_data.max()], density=False)
        super().__init__(column_data)

    def _compute_feature(self):
        """Calculates entropy"""
        # TODO timeit 
        # scipy_entr = feature.entropy_scipy(self.data_len, self.frequency)
        numpy_entr = feature.entropy_numpy(self.data_len, self.frequency)

        # print(scipy_entr, numpy_entr)
        return numpy_entr

    def get_params(self):
        # return self.feature, self.data_len, self.data_freq
        return {"feature": self.feature, "params": (self.frequency, self.bins)}

    def plt(self, **kwargs):
        if 'axs' in kwargs :
            kwargs["axs"].bar(self.bins[:-1], self.frequency, width=np.diff(self.bins), edgecolor="black", align="edge")
            kwargs["axs"].set_title(f"{self.column_name[0]}")
        else:
            ## Printar linha y com a distribuição
            plt.bar(self.bins[:-1], self.frequency, width=np.diff(self.bins), edgecolor="black", align="edge")
            plt.title(f"{self.column_name[0]}")
            # plt.stairs(self.frequency, self.bins, fill=True)

class BoxPlot(Viz):

    def __init__(self, column_data):
        self.data_len = len(column_data)
        self.params  = BoxPlot.calculate_boxplot_params(column_data)
        super().__init__(column_data)
    
    def calculate_boxplot_params(series: pd.Series) -> dict:
        """
        Calculate the boxplot parameters (min, Q1, median, Q3, max, and outliers) from a pandas Series.
        """
        q1 = series.quantile(0.25)
        median = series.median()
        q3 = series.quantile(0.75)
        iqr = q3 - q1  # Interquartile range
        whisker_low = q1 - 1.5 * iqr
        whisker_high = q3 + 1.5 * iqr
        
        # Calculate min and max within the whiskers
        min_val = series[series >= whisker_low].min()
        max_val = series[series <= whisker_high].max()
        
        # Identify outliers (fliers)
        outliers = series[(series < whisker_low) | (series > whisker_high)].values
    
        return {
            'whislo': whisker_low,   # Bottom whisker position
            'q1': q1,        # First quartile (25th percentile)
            'med': median,   # Median (50th percentile)
            'q3': q3,        # Third quartile (75th percentile)
            'whishi': whisker_high,   # Top whisker position
            'fliers': outliers  # Outliers
            }
        
    def _compute_feature(self):
        """Calculates """
        # pass
        return 0

    def get_params(self):
        # return self.feature, self.data_len, self.data_freq
        return {"feature": self.feature, **self.params}

    def plt(self, **kwargs):
        if 'axs' in kwargs :
            kwargs["axs"].bar(self.bins[:-1], self.frequency, width=np.diff(self.bins), edgecolor="black", align="edge")
        else:
            # plt.boxplot(column_data, patch_artist=True)  # fill with random  color
            fig, ax = plt.subplots()
            ax.bxp([self.params], showfliers=True, patch_artist=True, boxprops=dict(facecolor='lightblue'))
            ax.set_xticks([1], [self.column_name], rotation=45) # Rotaciona o rótulo do eixo x 
           
class Scatter(Viz):

    def __init__(self, data: pd.DataFrame):
        self.x, self.y = data.T.values
        super().__init__(data)

    def _compute_feature(self):
        """Calculates corr"""
        # TODO timeit 
        corr = pearsonr(self.x, self.y)
        return corr

    def get_params(self):
        return {"feature": self.feature, "params": None}

    def plt(self, **kwargs):
        if 'axs' in kwargs :
            kwargs["axs"].scatter(self.x, self.y)
            kwargs["axs"].set_title(f"{self.column_name[0]} x {self.column_name[1]}")
        else:
            ## Printar linha y com a distribuição
            plt.scatter(self.x, self.y)
            plt.title(f"{self.column_name[0]} x {self.column_name[1]}")