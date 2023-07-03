import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

df = pd.read_csv("energy.csv")

df

df.shape

df.info()

df.isnull().sum()

# убираем ненужный столбец и пропущенные значения
df = df.drop(['Unnamed: 0'], axis=1)
df.dropna(subset=['CO2_emission'], inplace=True)
df.dropna(subset=['Energy_consumption'], inplace=True)
df.dropna(subset=['Energy_production'], inplace=True)
df.dropna(subset=['GDP'], inplace=True)
df.dropna(subset=['Population'], inplace=True)
df.dropna(subset=['Energy_intensity_per_capita'], inplace=True)
df.dropna(subset=['Energy_intensity_by_GDP'], inplace=True)

# убираем nan
df = df.fillna(0)