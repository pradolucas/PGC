from itertools import combinations
from math import ceil

import matplotlib.pyplot as plt
import pandas as pd

from utils import viz
from utils.chart_types import BoxPlot, Histogram, Scatter


class VizSelector:
    """
    A class for selecting and managing different types of visualizations.

    Methods for each type of visualization are provided as class methods and a factory method (`create`) 
    is used to create the appropriate visualization based on the given type. The visualizations can be 
    sorted by feature and plotted in a grid layout.

    To add a new visualization:
    1. Create a subclass in `chart_types.py`.
    2. Define a class method to create that visualization in `VizSelector`.
    3. Map the visualization type to its class method in the `get_model_map` function.

    Attributes:
    -----------
    vizs : list
        A list of visualization objects (e.g., `Histogram`, `BoxPlot`, `Scatter`).
    ranked_vizs : Optional[list]
        A list of visualizations sorted by feature.

    Methods:
    --------
    get_model_map(cls) -> dict
        Returns a mapping of visualization types to their corresponding class methods.

    hist(cls, df: pd.DataFrame) -> 'VizSelector'
        Creates a `VizSelector` for histogram visualizations.

    box(cls, df: pd.DataFrame) -> 'VizSelector'
        Creates a `VizSelector` for box plot visualizations.

    scatter(cls, df: pd.DataFrame) -> 'VizSelector'
        Creates a `VizSelector` for scatter plot visualizations.

    create(cls, df: pd.DataFrame, viz_type: str) -> 'VizSelector'
        Factory method to choose and create the correct visualization type.

    get_rank(self) -> list
        Returns the visualizations sorted by feature.

    get_rank5(self) -> list
        Returns the top 5 visualizations based on feature.

    plt(self) -> None
        Plots the top 5 visualizations.

    plt_all(self, per_row: int = 5) -> None
        Plots all visualizations in a grid layout.
    """

    def __init__(self, vizs: viz):
        """
        Initializes the `VizSelector` with a list of visualization objects.

        Parameters:
        -----------
        vizs : list
            A list of visualization objects.
        """
        self.vizs = vizs
        self.ranked_vizs = None

    @classmethod
    def get_model_map(cls) -> dict:
        """
        Returns a mapping of visualization types to their corresponding class methods.

        Returns:
        --------
        dict
            A dictionary mapping visualization types to class methods.
        """
        model_map = {
            "hist": cls.hist,
            "scatter": cls.scatter,
            "box": cls.box,
        }
        return model_map

    @classmethod
    def hist(cls, df: pd.DataFrame) -> "VizSelector":
        """
        Creates a `VizSelector` for histogram visualizations.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data for the histogram.

        Returns:
        --------
        VizSelector
            A `VizSelector` object containing histogram visualizations.
        """
        # vizs = [Histogram(data) for _, data in df.items()]
        vizs = []
        for _, data in df.items():
            try:
                vizs.append(Histogram(data))
            except TypeError:
                pass
        return cls(vizs)

    @classmethod
    def box(cls, df: pd.DataFrame) -> "VizSelector":
        """
        Creates a `VizSelector` for box plot visualizations.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data for the box plot.

        Returns:
        --------
        VizSelector
            A `VizSelector` object containing box plot visualizations.
        """
        vizs = []
        for _, data in df.items():
            try:
                vizs.append(BoxPlot(data))
            except TypeError:
                pass
        return cls(vizs)

    @classmethod
    def scatter(cls, df: pd.DataFrame) -> "VizSelector":
        """
        Creates a `VizSelector` for scatter plot visualizations.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data for the scatter plots.

        Returns:
        --------
        VizSelector
            A `VizSelector` object containing scatter plot visualizations.
        """
        column_pairs = combinations(df.columns, 2)
        vizs = []
        for x, y in column_pairs:
            try:
                vizs.append(Scatter(df[[x, y]]))
            except TypeError:
                pass
        return cls(vizs)

    @classmethod
    def create(cls, df: pd.DataFrame, viz_type: str) -> "VizSelector":
        """
        Factory method to create the correct visualization type.

        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame containing the data for the visualization.
        viz_type : str
            The type of visualization to create.

        Returns:
        --------
        VizSelector
            A `VizSelector` object containing the selected type of visualizations.

        Raises:
        -------
        ValueError
            If the specified `viz_type` is not recognized.
        """
        model_map = cls.get_model_map()
        if viz_type not in model_map:
            raise ValueError(
                f"Unknown visualization type: {viz_type} not {set(model_map.keys())}"
            )
        return model_map[viz_type](df)

    def get_rank(self) -> list:
        """
        Returns the visualizations sorted by feature.

        Returns:
        --------
        list
            A sorted list of visualizations based on feature.
        """
        if not self.ranked_vizs:
            self.ranked_vizs = sorted(
                self.vizs, key=lambda x: abs(x.get_params()["feature"]), reverse=True
            )
        return self.ranked_vizs

    def get_rank5(self) -> list:
        """
        Returns the top 5 visualizations based on feature.

        Returns:
        --------
        list
            A list of the top 5 visualizations based on feature.
        """
        if not self.ranked_vizs:
            return self.get_rank()[:5]
        return self.ranked_vizs[:5]

    def plt(self) -> None:
        """
        Plots the top 5 visualizations.

        If there are less than 5 visualizations, it will plot them all.
        """
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
        plt.show()  # Disables stacked output from render_with_widgets

    def plt_all(self, per_row: int = 5) -> None:
        """
        Plots all visualizations in a grid layout.

        Parameters:
        -----------
        per_row : int, optional
            The number of visualizations per row (default is 5).
        """
        if not self.vizs:
            return

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
