from itertools import combinations
import pandas as pd
from utils.plots import *


class VizSelector():
    
    def __init__(self, df: pd.DataFrame, viz_type: str):
        self.vizs = []
        model_map = {"hist": ("1D", Histogram), "scatter": ("2D", Scatter)}
        dim, viz_model = model_map[viz_type]

        if(dim == '1D'):
            for _, data in df.items():
                self.vizs.append(viz_model(data))
        else:
            column_pairs = combinations(df.columns, 2)
            for x, y in column_pairs:
                self.vizs.append(viz_model(df[x], df[y]))
            
    def rank(self):
        vizs_sorted = sorted(self.vizs, key=lambda x: x.get_params()["feature"], reverse=True)
        return vizs_sorted
    
    def rank5(self):
        return self.rank()[:5]
    
    def plt(self):
        fig, axs = plt.subplots(1, 5, figsize=(20, 4))
        for idx, obj in enumerate(self.rank5()):
            axs[idx].scatter(obj.x, obj.y, color='blue')
            axs[idx].set_title(f"{obj.column_name[0]} x {obj.column_name[1]}")