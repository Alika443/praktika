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

#Годовое процентное изменение потребления энергии'
con0 = df[df['Country']=='World'][df['Energy_type']=='all_energy_types']
# изменение в процентах
temp_con0 = con0
temp_con0['процентное изменение'] = temp_con0['Energy_consumption'].pct_change() * 100

fig = px.area(temp_con0, x='Year', y='процентное изменение', title='Годовое процентное изменение потребления энергии')
fig.show()

# Ежегодное процентное изменение производства энергии
prod0 = df[df['Country']=='World'][df['Energy_type']=='all_energy_types']
# изменение в процентах
temp_prod0 = prod0
temp_prod0['процентное изменение'] = temp_prod0['Energy_production'].pct_change() * 100
fig = px.area(temp_prod0, x='Year', y='процентное изменение', title='Ежегодное процентное изменение производства энергии')



sample = df[df['Country']!='World'][df['Energy_type']!='all_energy_types']
years = sample['Year'].unique()
# среднее
list_m = []
for year in years:
    amount1 = sample[sample['Year']==year]['CO2_emission'].mean()
    list_m.extend([[year, amount1]])
temp_mean = pd.DataFrame(list_m, columns=['Year', 'co2_mean'])
temp_mean['co2_mean'] = round(temp_mean['co2_mean'], 2)

# стандартное отклонение
list_sd = []
for year in years:
    amount = sample[sample['Year']==year]['CO2_emission'].std()
    list_sd.extend([[year, amount]])
temp_sd = pd.DataFrame(list_sd, columns=['Year', 'co2_sd'])
temp_sd['co2_sd'] = round(temp_sd['co2_sd'], 2)

# графики
fig = make_subplots(rows=2, cols=1)
fig.add_trace(
    go.Scatter(x=temp_mean['Year'], y=temp_mean['co2_mean'], mode="lines+markers"),
    row=1, col=1
)
fig.add_trace(
    go.Scatter(x=temp_sd['Year'], y=temp_sd['co2_sd'], mode="lines+markers"),
    row=2, col=1
)
fig.update_yaxes(title_text="среднее", row=1, col=1)
fig.update_yaxes(title_text="стандартное отклонение", row=2, col=1)
fig.update_traces(textposition="bottom center")
fig.update_layout(height=700, width=900, title_text="Среднегодовой и стандартное отклонение выбросов CO2 в каждой стране")
fig.show()

# Выбросы CO2 в каждой стране с 1988 по 2019 год
cd2 = df[df['Country']!='World'][df['Energy_type']=='all_energy_types']
temp_cd2 = cd2[['Country', 'Year', 'CO2_emission']].groupby(['Country','Year']).sum().reset_index()

px.choropleth(data_frame=temp_cd2, locations="Country", locationmode='country names', animation_frame="Year",
              color='CO2_emission', title="Выбросы CO2 в каждой стране с 1988 по 2019 год")

# производство энергии в каждой стране с 1988 по 2019 год
cd2 = df[df['Country']!='World'][df['Energy_type']=='all_energy_types']

temp_cd2 = cd2[['Country', 'Year', 'Energy_production']].groupby(['Country','Year']).sum().reset_index()

px.choropleth(data_frame=temp_cd2, locations="Country", locationmode='country names', animation_frame="Year",
              color='Energy_production', title="производство энергии в каждой стране с 1988 по 2019 год")

# процентное изменение потребление разных видов энергии
con0 = df[df['Country']=='World'][['Year','Country', 'Energy_type', 'Energy_consumption']]

# уголь
coal = con0
coal['проц. измен.'] = con0[con0['Energy_type']=='coal']['Energy_consumption'].pct_change() * 100
coal = coal[coal['проц. измен.'].notna()]

# природный газ
nat_gas = con0
nat_gas['проц. измен.'] = con0[con0['Energy_type']=='natural_gas']['Energy_consumption'].pct_change() * 100
nat_gas = nat_gas[nat_gas['проц. измен.'].notna()]

# нефть
pet_oth = con0
pet_oth['проц. измен.'] = con0[con0['Energy_type']=='petroleum_n_other_liquids']['Energy_consumption'].pct_change() * 100
pet_oth = pet_oth[pet_oth['проц. измен.'].notna()]

# ядераная энергия
nuclear = con0
nuclear['проц. измен.'] = con0[con0['Energy_type']=='nuclear']['Energy_consumption'].pct_change() * 100
nuclear = nuclear[nuclear['проц. измен.'].notna()]

# возобновляемые источники энергии
ren_oth = con0
ren_oth['проц. измен.'] = con0[con0['Energy_type']=='renewables_n_other']['Energy_consumption'].pct_change() * 100
ren_oth = ren_oth[ren_oth['проц. измен.'].notna()]

final_df = pd.concat([coal, nat_gas, pet_oth, nuclear, ren_oth], axis=0)

fig = px.area(final_df, x='Year', y='проц. измен.', facet_col='Energy_type', color='Energy_type', facet_col_wrap=2,
             title='Годовой процентный показатель увеличения/уменьшения потребления каждого вида энергии')
fig.show()


# топ 20 стран - крупнейших потребителей энергии
con1 = df[df['Country']!='World'][df['Energy_type']=='all_energy_types']

list_con = []

# Сумма потребления всех видов энергии за все года
for country in con1['Country'].unique():
    total_con = con1[con1['Country']==country]['Energy_consumption'].sum(axis=0)
    list_con.extend([[country, total_con]])

# Временный набор данных по всем странам и их соответствующему общему потреблению за определенный период времени
top_con = pd.DataFrame(list_con, columns=['Country', 'общее потребление']).sort_values(by='общее потребление',ascending=False)

# Plotting the top 20 Consumers
fig = px.bar(top_con.head(20), x='Country', y='общее потребление', title='топ 20 стран - крупнейших потребителей энергии')
fig.show()

#Топ-20 стран по выбросу CO2
cd1 = df[df['Country']!='World']

list = []

for country in cd1['Country'].unique():
    total = cd1[cd1['Country']==country]['CO2_emission'].sum(axis=0)
    list.extend([[country, total]])

# Временный набор данных по всем странам и их соответствующим общим выбросам CO2 за определенный период времени
temp_cd = pd.DataFrame(list, columns=['Country', 'Total_CO2']).sort_values(by='Total_CO2',ascending=False)

fig = px.bar(temp_cd.head(20), x='Country', y='Total_CO2', title='Топ-20 стран по выбросу CO2')
fig.show()


# страны по кол-ву потребления отдельных видов энергии
fig, ax = plt.subplots(2, 3, figsize = (20, 10))

types = ['coal', 'natural_gas', 'petroleum_n_other_liquids', 'nuclear', 'renewables_n_other']

con3 = df[df['Country']!='World'].reset_index()

for idx, (type, axes) in enumerate(zip(types, ax.flatten())):

    dirty = con3[con3['Energy_type']==type].groupby(['Country'])['Energy_consumption'].sum().reset_index().sort_values(by='Energy_consumption', ascending=False)

    sns.barplot(ax=axes, data=dirty.head(6), x='Country', y='Energy_consumption', palette="Blues_d")

    axes.set_title(f'{type}')

    for tick in axes.get_xticklabels():
        tick.set_rotation(35)


plt.tight_layout(pad=0.2, w_pad=3, h_pad=3)
plt.show()


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import datetime as dt
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff

df = pd.read_csv("energy.csv")
df
df.info()


# Разделение набора данных на два датафрейма
df = pd.read_csv("energy.csv")
df=df.fillna(0)
df=df.dropna(0)
df=df.dropna(how='any')

#Удаление лишних столбцов
df.drop (columns=df.columns [10], axis= 1 , inplace= True )
df.drop (columns=df.columns [4], axis= 1 , inplace= True )
df.drop (columns=df.columns [2], axis= 1 , inplace= True )
df. drop (columns=df. columns [1], axis= 1 , inplace= True )
df.drop (columns=df.columns [0], axis= 1 , inplace= True )
df['Year'] = df['Year'].astype (float)

df_1_to_11 = df[df['Year'].isin(range(2000, 2018))] # Выбор строк с годами от 2000 до 2018
df_12 = df[df['Year'] == 2019] # Выбор строк с 2019 годом

# Вывод первых 5 строк каждого датафрейма
df_1_to_11.to_csv("1to11.csv", index=False)
df_12.to_csv("to12.csv", index=False)
print(df_1_to_11.head())
print(df_12.head())


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import tensorflow as tf

df_1_to_11 = df_1_to_11.drop('Year', axis = 1)

# Разделение набора данных на обучающиеся и тестовые наборы
train_df, test_df = train_test_split(df_1_to_11, test_size=0.2, random_state=50)

# Преобразование обучающих и тестовых данных в 3D-массив для ввода в LSTM
X_train = train_df.drop("Energy_production", axis=1).values.reshape(-1, 1, 4)
X_test = test_df.drop("Energy_production", axis=1).values.reshape(-1, 1, 4)
y_train = train_df["Energy_production"].values.reshape(-1, 1)
y_test = test_df["Energy_production"].values.reshape(-1, 1)
df


# Создание модели LSTM
model = Sequential()
model.add(LSTM(16, input_shape=(1, 4)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()


model.fit(X_train, y_train, epochs=20, verbose=0)

# предсказание на данных X_train
preds = model.predict(X_test)

df_12=df_12.dropna(how='any')


next_day = df_12[["GDP", "Population", "Energy_intensity_per_capita","Energy_intensity_by_GDP"]].values.reshape(-1, 1, 4)
# Делаем прогнозы, используя модель
next_day_preds = model.predict(next_day)

# Добавляем прогнозируемые значения в исходный фрейм данных
df_12["Energy_production"] = next_day_preds.flatten()

# Записываем обновленные данные в тот же файл excel
df_12.to_csv("PredictData.csv", index=False)