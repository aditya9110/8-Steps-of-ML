import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('time_series_covid19_confirmed_global.csv')
df = df.drop(columns=['Province/State', 'Long', 'Lat'])
removal = []
for column in df:
    content = df[column]
    if list(content).count(0) * 100 / content.count() >= 70:
        removal.append(column)
df = df.drop(columns=removal)
df = df.groupby(['Country/Region']).sum()
df.reset_index(inplace=True)
# print(df)

df1 = df[['Country/Region', '3/24/20']]
df1 = df1[df1['3/24/20'] > 200]
# print(df1)
num_cases = list(df1['3/24/20'])
countries = list(df1['Country/Region'])

plt.style.use('fivethirtyeight')
y_pos = np.arange(len(countries))
plt.barh(y_pos, num_cases, align='center')
plt.yticks(ticks=y_pos, labels=countries, fontsize=7)
plt.xlabel('No. of confirmed cases')
plt.ylabel('Countries')
plt.gca().invert_yaxis()
plt.title('COVID-19 cases (Countries-wise on 24 Mar 2020)')
plt.grid(True)
plt.show()

df2 = df.loc[[78]]
df2.set_index('Country/Region', inplace=True)
# print(df2)
num_cases = list(df2.columns)
# print(num_cases)
dates = list(df.iloc[78, :])
dates = dates[1:]
# print(dates)

plt.style.use('fivethirtyeight')
plt.plot(num_cases, dates, 'o-b')
plt.xticks(fontsize=7, rotation=90)
plt.xlabel('Dates')
plt.ylabel('No. of Cases')
plt.title('COVID-19 Cases in India')
plt.grid(True)
plt.tight_layout()
plt.show()

df3 = pd.read_csv('continents.csv')
df3.rename(columns={'Country': 'Country/Region'}, inplace=True)
df3 = pd.merge(df, df3, on='Country/Region', how='inner')
df3 = df3.groupby(['Continent']).sum()
df3.reset_index(inplace=True)
df3 = df3[['Continent', '3/24/20']]
# print(df3)

slices = list(df3['3/24/20'])
labels = list(df3['Continent'])
colors = ['#C70039', '#70B2FF', '#FFC300', '#DAF7A6', '#FF5733']
plt.pie(slices, labels=labels, colors=colors, autopct='%1.1f%%',
        wedgeprops={'edgecolor': 'black', 'linewidth': 1})
plt.title('COVID-19 Cases along Continents')
plt.grid(True)
plt.tight_layout()
plt.show()
