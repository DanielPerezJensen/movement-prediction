import numpy as np
import ast
import pandas as pd

data = []

# Gather raw data as list of tuples
with open("data/data.txt", "r") as f:
    for line in f:
        try:
            data.append([ast.literal_eval(s) for s in
                         line.strip().split(";")])
        except SyntaxError:
            pass

ns = [1, 2, 3, 4, 5]

for n in ns:
    print(n)
    # Convert list of tuples to dataframe and save it as a csv
    path_data = []
    for path in data:
        if len(path) > 2 + n:
            for i in range(len(path[:-2 - n])):
                x0, y0 = path[i]
                x1, y1 = path[i + 1]
                x2, y2 = path[i + 2]
                x5, y5 = path[i + 2 + n]
                path_datapoint = [x0, y0, x1, y1, x2, y2, x5, y5]
                path_data.append(path_datapoint)

    path_df = pd.DataFrame(path_data,
                           columns=["x0", "y0", "x1", "y1", "x2", "y2",
                                    "x3", "y3"])
    path_df.to_csv(f"data/preprocessed_data-n={n}.csv", sep="\t")
