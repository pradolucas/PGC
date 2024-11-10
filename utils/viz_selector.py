from itertools import combinations
import pandas as pd
from utils import viz
from utils.chart_types import Histogram, Scatter, BoxPlot
import matplotlib.pyplot as plt
from math import ceil


class VizSelector:

    def __init__(self, vizs: viz):
        """Initialize with a list of visualization objects"""
        self.vizs = vizs
        self.rank5 = None

    @classmethod
    def hist(cls, df: pd.DataFrame):
        """Create VizSelector for 1D visualizations (Histograms)"""
        # vizs = [Histogram(data) for _, data in df.items()]
        vizs = []
        for _, data in df.items():
            try:
                vizs.append(Histogram(data))
            except TypeError:
                pass
        return cls(vizs)

    @classmethod
    def box(cls, df: pd.DataFrame):
        """Create VizSelector for 1D visualizations (Box plot)"""
        # vizs = [BoxPlot(data) for _, data in df.items()]
        vizs = []
        for _, data in df.items():
            try:
                vizs.append(BoxPlot(data))
            except TypeError:
                pass
        return cls(vizs)

    @classmethod
    def scatter(cls, df: pd.DataFrame):
        """Create VizSelector for 2D visualizations (Scatter plots)"""
        column_pairs = combinations(df.columns, 2)
        # vizs = [Scatter(df[[x, y]]) for x, y in column_pairs]
        vizs = []
        for x, y in column_pairs:
            try:
                vizs.append(Scatter(df[[x, y]]))
            except TypeError:
                pass
        return cls(vizs)

    @classmethod
    def create(cls, df: pd.DataFrame, viz_type: str):
        """Factory method to choose correct visualization type"""
        model_map = {"hist": cls.hist, "scatter": cls.scatter, "box": cls.box}
        if viz_type not in model_map:
            raise ValueError(f"Unknown visualization type: {viz_type}")
        return model_map[viz_type](df)

    def get_rank(self):
        vizs_sorted = sorted(
            self.vizs, key=lambda x: abs(x.get_params()["feature"]), reverse=True
        )
        return vizs_sorted

    def get_rank5(self):
        if self.rank5:
            return self.rank5
        return self.get_rank()[:5]

    def plt(self):
        if not self.vizs:
            return

        rank5 = self.get_rank5()
        n_vizs = len(rank5)
        n_cols = min(n_vizs, 5)  # for cases with less vizs than per row default value
        _, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
        axs = (
            axs.flatten() if n_vizs > 1 else [axs]
        )  # Make axs iterable if there's only one plot

        for idx, obj in enumerate(rank5):
            obj.plt(axs=axs[idx], title_idx=idx)
            
        plt.tight_layout()
        # plt.show() ## Apaga o output de interactive_dataframe

    def plt_all(self, per_row=5):
        if not self.vizs:
            return

        # vizs = self.get_rank()

        # Calculate the total number of Viz objects
        n_vizs = len(self.vizs)

        # Determine the grid size
        n_rows = ceil(n_vizs / per_row)
        n_cols = min(
            n_vizs, per_row
        )  # for cases with less vizs than per row default value

        # Create subplots and handle single plot case
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axs = (
            axs.flatten() if n_vizs > 1 else [axs]
        )  # Make axs iterable if there's only one plot

        # Plot each Viz object in the grid
        idx = 0
        for viz_obj in self.vizs:
            viz_obj.plt(axs=axs[idx])
            idx += 1

        # Remove any unused subplots
        for i in range(idx, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.show()

        # for row_idx in range(n_rows):
        #     for col_idx in range(per_row):
        #         idx_viz = 5 * row_idx + col_idx
        #         if idx_viz >= n_vizs:
        #             break
        #         obj = self.vizs[idx_viz]
        #         obj.plt(axs=axs[row_idx][col_idx], title_idx=idx_viz)


## Para inserir uma viz nova:
## 1. Create a viz-child class in chart_types
## 2. Create a classmethod for the class creation of this viz on VizSelector
## 3. Map viz creation in create classmethod
