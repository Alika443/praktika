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

# кол-во выбросов со2 в каждом году
plt.figure(figsize=(18,4))
color = plt.cm.copper(np.linspace(0, 1, 10))
df.groupby(['Year'])['CO2_emission'].count().plot(kind='bar', title = "кол-во выбросов со2 в каждом году", width=.4,color=color);
plt.xticks(rotation=45);

# кол-во выбросов со2 всвязи с типом энергии
plt.figure(figsize=(18,4))
color = plt.cm.copper(np.linspace(0, 1, 10))
df.groupby(['Energy_type'])['CO2_emission'].count().plot(kind='bar', title = "кол-во выбросов со2 всвязи с типом энергии", width=.4,color=color);
plt.xticks(rotation=45);

# энергия, выработанная в каждой стране
fig = px.box(df,
        x="Country",
        y="Energy_production",
        title = "энергия, выработанная в каждой стране",
        labels = {"x" : "Дни"})

fig.update_traces(width=0.5)
fig.show()

# Выработка энергии в год
fig = px.bar(df,
              x="Year",
              y="Energy_production",
              color = "Energy_production",
              title="Выработка энергии в год")
fig.update_traces(width=0.6)
fig.update_layout(barmode='group', xaxis_tickangle=-45)
fig.show()

# потребление энергии каждого вида энергии в год
f, axes = plt.subplots(1,1, figsize = (11,5.5))
for a,(b,c) in enumerate(df.groupby('Energy_type')):
    axes.scatter(c.Year, c.Energy_consumption, label = b)
axes.legend()
axes.grid(True)
plt.title('потребление энергии каждого вида энергии в год')
plt.xlabel('год')
plt.show()

# распределение типов энергии по кол-ву выделения со2
temp_dist = df.groupby('Energy_type').count()['CO2_emission'].reset_index().sort_values(by='CO2_emission',ascending=False)
temp_dist.style.background_gradient(cmap='spring')

# распределение типов энергии по кол-ву выделения со2 в виде круговой диаграммы
percent = temp_dist['CO2_emission']
labels= temp_dist['Energy_type']
fig = go.Figure()
fig.add_trace(go.Pie(values=percent, labels=labels))
fig.show()