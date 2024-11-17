import pickle
from datetime import datetime
from math import ceil
from typing import Optional

import pandas as pd
from IPython.display import display
from ipywidgets import VBox, widgets
from matplotlib import pyplot as plt

from utils.viz_selector import VizSelector


class InteractiveDataFrame(pd.DataFrame):
    """
    A subclass of pandas DataFrame that adds interactive plotting functionality.

    Attributes:
    -----------
    selected_plot : str
        The type of plot currently selected (default is "hist").
    pltd : dict
        A dictionary to store plotted visualizations, with plot types as keys and lists of visualizations as values.

    Methods:
    --------
    get_pltd(self) -> dict
        Returns the dictionary of plotted visualizations.

    get_viz(self, selector_dict: dict) -> dict
        Retrieves selected visualizations based on the provided selector dictionary.

    save_plt(self, selector_plt: dict, fname: Optional[str] = None) -> None
        Saves selected visualizations to a pickle file.

    load_plt(fname: str) -> dict
        Loads visualizations from a pickle file.

    plt5(self, viz_type: str) -> None
        Plots the top 5 visualizations for a given plot type.

    plt_all(self, viz_type: str) -> None
        Plots all visualizations for a given plot type.

    plt_from_selector(selector_dict: dict, per_row: int = 3) -> None
        Plots visualizations in a grid layout based on the given selector dictionary.

    render_with_widgets(self) -> None
        Renders the DataFrame with interactive widgets for plot selection and display.
    """

    _metadata = ["selected_plot", "pltd"]  # Custom attributes metadata

    def __init__(self, *args, **kwargs):
        """
        Initializes the DataFrame with additional custom attributes: selected_plot and pltd.

        Parameters:
        -----------
        args: Additional positional arguments passed to the pandas DataFrame constructor.
        kwargs: Additional keyword arguments passed to the pandas DataFrame constructor.
        """
        super().__init__(*args, **kwargs)
        self.selected_plot = "hist"  # Default plot type
        self.pltd = {}  # To store plotted visualizations

    def get_pltd(self) -> dict:
        """
        Returns the dictionary of plotted visualizations.

        Returns:
        --------
        dict
            The dictionary containing plotted visualizations for each plot type.
        """
        return self.pltd

    def get_viz(self, selector_dict: dict) -> dict:
        """
        Retrieves selected visualizations based on the provided selector dictionary.

        Parameters:
        -----------
        selector_dict : dict
            A dictionary where keys are plot types and values are lists of indices specifying which visualizations to retrieve.

        Returns:
        --------
        dict
            A dictionary of selected visualizations based on the provided selector dictionary.
        """
        res = {}
        for viz_type, idxs in selector_dict.items():
            if viz_type in self.pltd:
                res[viz_type] = [self.pltd[viz_type][idx] for idx in idxs]
        return res

    def save_plt(self, selector_plt: dict, fname: Optional[str] = None) -> None:
        """
        Saves selected visualizations to a pickle file.

        Parameters:
        -----------
        selector_plt : dict
            A dictionary of visualizations to save.
        fname : Optional[str], optional
            The filename to save the visualizations to. If not provided, a default filename is generated.
        """
        if not fname:
            formatted_datetime = datetime.now().strftime("%y%m%d_%H%M")
            fname = f"plt_selection_{formatted_datetime}.pickle"

        selector_plt = self.get_viz(selector_plt)
        with open(fname, "wb") as f:
            pickle.dump(selector_plt, f)

    @staticmethod
    def load_plt(fname: str) -> dict:
        """
        Loads visualizations from a pickle file.

        Parameters:
        -----------
        fname : str
            The filename from which to load the visualizations.

        Returns:
        --------
        dict
            A dictionary of loaded visualizations.
        """
        with open(fname, "rb") as f:
            file = pickle.load(f)
            return file

    def plt5(self, viz_type: str) -> None:
        """
        Plots the top 5 visualizations for a given plot type.

        Parameters:
        -----------
        viz_type : str
            The type of visualization to plot.

        Raises:
        -------
        ValueError
            If the visualization type is not valid.
        """
        model_map = VizSelector.get_model_map().keys()
        if viz_type not in model_map:
            raise ValueError(
                f"Unknown visualization type: {viz_type} not {set(model_map)}"
            )
        VizSelector.create(self, viz_type).plt_all()

    def plt_all(self, viz_type: str) -> None:
        """
        Plots all visualizations for a given plot type.

        Parameters:
        -----------
        viz_type : str
            The type of visualization to plot.

        Raises:
        -------
        ValueError
            If the visualization type is not valid.
        """
        model_map = VizSelector.get_model_map().keys()
        if viz_type not in model_map:
            raise ValueError(
                f"Unknown visualization type: {viz_type} not {set(model_map)}"
            )
        VizSelector.create(self, viz_type).plt_all()

    @staticmethod
    def plt_from_selector(selector_dict: dict, per_row: int = 3) -> None:
        """
        Plots visualizations in a grid layout based on the given selector dictionary.

        Parameters:
        -----------
        selector_dict : dict
            A dictionary where keys are plot types and values are lists of visualization objects.
        per_row : int, optional
            The number of visualizations per row in the grid layout (default is 3).
        """
        if not selector_dict:
            return

        # Calculate the total number of Viz objects
        n_vizs = sum(len(viz_objs) for viz_objs in selector_dict.values())

        # Determine the grid size
        n_rows = ceil(n_vizs / per_row)
        n_cols = min(n_vizs, per_row)

        # Create subplots and handle single plot case
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
        axs = (
            axs.flatten() if n_vizs > 1 else [axs]
        )  # Make axs iterable if there's only one plot

        # Plot each Viz object in the grid
        idx = 0
        for viz_objs in selector_dict.values():
            for viz_obj in viz_objs:
                viz_obj.plt(axs=axs[idx])
                idx += 1

        # Remove any unused subplots
        for i in range(idx, len(axs)):
            fig.delaxes(axs[i])

        plt.tight_layout()
        plt.show()

    def render_with_widgets(self) -> None:
        """
        Renders the DataFrame with interactive widgets for plot selection and display.
        Displays a dropdown for plot type selection, a button to generate the selected plot,
        and an HTML representation of the first 10 rows of the DataFrame.
        """
        drop_down_options = VizSelector.get_model_map().keys()
        dropdown = widgets.Dropdown(
            options=drop_down_options,
            value=self.selected_plot,
            description="Plot type:",
            disabled=False,
        )

        # Button to generate the selected plot
        button = widgets.Button(description="Generate Plot")

        # Output widget to display the plot
        output = widgets.Output()

        # Convert DataFrame to an HTML table with styling
        html_df = self.head(10).to_html(index=False, classes="dataframe", border=0)
        css = """
        <style>
        .dataframe {
            border-collapse: collapse;
            width: 100%;
        }
        .dataframe th, .dataframe td {
            border: 1px solid #ddd;
            padding: 8px;
        }
        .dataframe th {
            background-color: #f2f2f2;
            text-align: center;
        }
        </style>
        """

        # Combine CSS and HTML table
        html_content = widgets.HTML(value=css + html_df)

        # Function to handle button click event
        def on_button_click(b):
            self.selected_plot = dropdown.value  # Update selected plot type
            output.clear_output()  # Clear previous output

            with output:
                # Generate plot using VizSelector
                if self.selected_plot not in self.pltd:
                    obj = VizSelector.create(self, self.selected_plot)
                    top5 = obj.get_rank5()  # Get top 5 visualizations
                    self.pltd[self.selected_plot] = top5
                    obj.plt()  # Generate the plot
                else:
                    top5 = self.pltd[self.selected_plot]
                    n_vizs = len(top5)
                    n_cols = min(
                        n_vizs, 5
                    )  # for cases with less vizs than per row default value
                    _, axs = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
                    axs = (
                        axs.flatten() if n_vizs > 1 else [axs]
                    )  # Make axs iterable if there's only one plot

                    for idx, obj in enumerate(top5):
                        obj.plt(axs=axs[idx], title_idx=idx)

                    plt.tight_layout()
                    plt.show()

                display(
                    f"Selected Plot: {self.selected_plot}"
                )  # Display selected plot type

        # Bind the button click event to the function
        button.on_click(on_button_click)

        # Display widgets
        display(VBox([dropdown, html_content, button, output]))

    # def _repr_html_(self):
    #     # Calls render_with_widgets to display the dataframe with widgets in Jupyter
    #     self._render_with_widgets()
    #     return ""  # Return empty string as render_with_widgets handles display
