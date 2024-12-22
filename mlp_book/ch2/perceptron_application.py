import numpy as np
import pandas as pd

df = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                header = None, encoding = "utf-8")
print(df.head())
print(len(df))
