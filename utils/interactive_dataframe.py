import pandas as pd
import pickle
from math import ceil
from ipywidgets import widgets, VBox
from matplotlib import pyplot as plt
from IPython.display import display
from utils.viz_selector import VizSelector
from datetime import datetime


class InteractiveDataFrame(pd.DataFrame):
    _metadata = ["selected_plot", "pltd"]  # Custom attributes metadata

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selected_plot = "hist"  # Default plot type
        self.pltd = {}  # To store plotted visualizations

    def get_pltd(self):
        return self.pltd

    def get_viz(self, selector_dict: dict):
        res = {}
        for viz_type, idxs in selector_dict.items():
            if viz_type in self.pltd:
                res[viz_type] = [self.pltd[viz_type][idx] for idx in idxs]
        return res

    def save_plts(self, selector_plt, fname=None):
        if not fname:
            formatted_datetime = datetime.now().strftime("%y%m%d_%H%M")
            fname = f"plt_selection_{formatted_datetime}.pickle"

        selector_plt = self.get_viz(selector_plt)
        with open(fname, "wb") as f:
            pickle.dump(selector_plt, f)

    @staticmethod
    def load_plts(fname):
        with open(fname, "rb") as f:
            file = pickle.load(f)
            return file

    # TODO
    # def plt5(self, viz_type):
    #     VizSelector.create(self, viz_type).plt()

    def plt_all(self, viz_type):
        VizSelector.create(self, viz_type).plt_all()

    @staticmethod
    def plt_from_selector(selector_dict: dict, per_row=3):
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

    def render_with_widgets(self):
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
                    n_cols = min(n_vizs, 5)  # for cases with less vizs than per row default value
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
