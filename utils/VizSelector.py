from itertools import combinations
import pandas as pd
from utils import Viz
from utils.chart_types import Histogram, Scatter
import matplotlib.pyplot as plt


class VizSelector():

    def __init__(self, vizs: Viz):
        """Initialize with a list of visualization objects"""
        self.vizs = vizs

    @classmethod
    def hist(cls, df: pd.DataFrame):
        """Create VizSelector for 1D visualizations (e.g., Histograms)"""
        vizs = [Histogram(data) for _, data in df.items()]
        return cls(vizs)

    @classmethod
    def scatter(cls, df: pd.DataFrame):
        """Create VizSelector for 2D visualizations (e.g., Scatter plots)"""
        column_pairs = combinations(df.columns, 2)
        vizs = [Scatter(df[[x, y]]) for x, y in column_pairs]
        return cls(vizs)

    @classmethod
    def create(cls, df: pd.DataFrame, viz_type: str):
        """Factory method to choose correct visualization type"""
        model_map = {
            "hist": cls.hist,
            "scatter": cls.scatter
        }
        if viz_type not in model_map:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        return model_map[viz_type](df)
    
    def rank(self):
        vizs_sorted = sorted(self.vizs, key=lambda x: x.get_params()["feature"], reverse=True)
        return vizs_sorted
    
    def rank5(self):
        return self.rank()[:5]
    
    def plt(self):
        _, axs = plt.subplots(1, 5, figsize=(20, 4))
        for idx, obj in enumerate(self.rank5()):
            obj.plt(axs=axs[idx])