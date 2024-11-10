from itertools import combinations
import pandas as pd
from utils import Viz
from utils.chart_types import Histogram, Scatter, BoxPlot
import matplotlib.pyplot as plt
from math import ceil


class VizSelector:

    def __init__(self, vizs: Viz):
        """Initialize with a list of visualization objects"""
        self.vizs = vizs

    @classmethod
    def hist(cls, df: pd.DataFrame):
        """Create VizSelector for 1D visualizations (Histograms)"""
        vizs = [Histogram(data) for _, data in df.items()]
        return cls(vizs)

    @classmethod
    def box(cls, df: pd.DataFrame):
        """Create VizSelector for 1D visualizations (Box plot)"""
        vizs = [BoxPlot(data) for _, data in df.items()]
        return cls(vizs)

    @classmethod
    def scatter(cls, df: pd.DataFrame):
        """Create VizSelector for 2D visualizations (Scatter plots)"""
        column_pairs = combinations(df.columns, 2)
        vizs = [Scatter(df[[x, y]]) for x, y in column_pairs]
        return cls(vizs)

    @classmethod
    def create(cls, df: pd.DataFrame, viz_type: str):
        """Factory method to choose correct visualization type"""
        model_map = {"hist": cls.hist, "scatter": cls.scatter, "box": cls.box}
        if viz_type not in model_map:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        return model_map[viz_type](df)

    def rank(self):
        vizs_sorted = sorted(
            self.vizs, key=lambda x: abs(x.get_params()["feature"]), reverse=True
        )
        return vizs_sorted

    def rank5(self):
        return self.rank()[:5]

    def plt(self):
        _, axs = plt.subplots(1, 5, figsize=(20, 4))
        for idx, obj in enumerate(self.rank5()):
            obj.plt(axs=axs[idx], title_idx=idx)

    def plt_all(self):
        n_vizs = len(self.vizs)
        per_row = 5
        n_rows = ceil(n_vizs / per_row)
        _, axs = plt.subplots(n_rows, per_row, figsize=(4 * per_row, 4 * n_rows))
        for row_idx in range(n_rows):
            for col_idx in range(per_row):
                idx_viz = 5 * row_idx + col_idx
                if idx_viz >= n_vizs:
                    break
                obj = self.vizs[idx_viz]
                obj.plt(axs=axs[row_idx][col_idx], title_idx=idx_viz)


## Para inserir uma viz nova:
## 1. Create a viz-child class in chart_types
## 2. Create a classmethod for the class creation of this viz on VizSelector
## 3. Map viz creation in create classmethod
