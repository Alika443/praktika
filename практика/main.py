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

# стандартный коэффициент корреляции
df_corr = df.corr()
df_corr

# цветовая карта распределения величин
x = list(df_corr.columns)
y = list(df_corr.index)
z = np.array(df_corr)

fig = ff.create_annotated_heatmap(x = x,
                                  y = y,
                                  z = z,
                                  annotation_text = np.around(z, decimals=2))
fig.show()

# взаимосвязь всех данных датасета cо всеми видами энергии
temp_pp = df[df['Country']=='World'][df['Energy_type']!='all']

sns.pairplot(temp_pp, hue='Energy_type',palette="inferno")
plt.show()

# совокупность разных графиков
with plt.rc_context(rc = {'figure.dpi': 250, 'axes.labelsize': 9,
                          'xtick.labelsize': 10, 'ytick.labelsize': 10,
                          'legend.title_fontsize': 7, 'axes.titlesize': 12,
                          'axes.titlepad': 7}):

    con = df[df['Country']=='World']

    fig_4, ax_4 = plt.subplots(2, 2, figsize = (10, 8), gridspec_kw = {'width_ratios': [3, 3], 'height_ratios': [3, 4]})
    ax_flat = ax_4.flatten()

    sns.lineplot(ax=ax_flat[0], data=con[con['Energy_type']!='all'], x='Year', y='Energy_consumption').set_title('потребление всех видов энергии во всем мире')

    sns.lineplot(ax=ax_flat[1], data=con[con['Energy_type']!='all'], x='CO2_emission', y='Energy_consumption',lw=3).set_title('Взаимосвязь потребления энергии и выброса CO2')

    sns.lineplot(ax=ax_flat[2], data=con[con['Energy_type']!='all'], x='Year', y='Energy_consumption', hue='Energy_type',lw=3).set_title('Ежегодное потребление каждого вида энергии в мире')

    sns.stripplot(ax=ax_flat[3], data=con[con['Energy_type']!='all'],x='Energy_type', y='Energy_consumption', jitter=.3, linewidth=1).set_title('потребление по разным видам энергии во всем мире')

    ax_flat[3].tick_params(axis='x', rotation=55)

    plt.tight_layout(pad = 1)
    plt.show()