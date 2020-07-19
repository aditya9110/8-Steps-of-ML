import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.options.display.max_columns = 20
df = pd.read_csv('Titanic training data.csv')
df = df.drop(columns=['Name', 'Ticket', 'Fare'])

df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

random_ages = np.random.randint(df['Age'].mean() - df['Age'].std(),
                                df['Age'].mean() + df['Age'].std())
random_emb = np.random.choice(['S', 'C', 'Q'])
df.fillna(value={'Age': random_ages, 'Embarked': random_emb}, inplace=True)


def age_classify(x):
    if x < 20:
        return 0
    elif 20 <= x < 40:
        return 1
    elif 40 <= x < 60:
        return 2
    elif 60 <= x < 80:
        return 3
    elif 80 <= x:
        return 4


df['Age Range'] = df['Age'].apply(age_classify)
df = df.drop(columns='Age')

df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

df['Cabin'] = df['Cabin'].apply(lambda x: 0 if type(x) == float else 1)  # if has cabin then 1 else 0

df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
df['FamilySize'] = df['FamilySize'].apply(lambda x: 0 if x == 1 else (1 if 1 < x < 6 else 2))

# print(df)

# sex vs survived graph
plt.style.use('fivethirtyeight')
gender = ['Male', 'Female']
x = np.arange(len(gender))
df_Sex = pd.crosstab(df.Sex, df.Survived)
survived = list(df_Sex[1])
deaths = list(df_Sex[0])
w = 0.30
plt.bar(x - w / 2, survived, label='Survived', color='g', width=w)
plt.bar(x + w / 2, deaths, label='Deaths', color='r', width=w)
plt.xticks(ticks=x, labels=gender)
plt.ylabel('Number of people')
plt.title('Survival Gender based')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# age vs survived graph
age_grps = ['0-19', '20-39', '40-59', '60-79', '80+']
plt.style.use('fivethirtyeight')
x = np.arange(len(age_grps))
df_age = pd.crosstab(df['Age Range'], df.Survived)
survived = list(df_age[1])
deaths = list(df_age[0])
# print(survived, deaths)
w = 0.30
plt.bar(x - w / 2, survived, label='Survived', color='g', width=w)
plt.bar(x + w / 2, deaths, label='Deaths', color='r', width=w)
plt.title('Survival Age based')
plt.xticks(ticks=x, labels=age_grps)
plt.ylabel('Number of people')
plt.legend()
plt.tight_layout()
plt.grid(True)
plt.show()

# familysize vs survival graph
plt.style.use('fivethirtyeight')
family = ['Alone', 'Small', 'Big']
x = np.arange(len(family))
df_family = pd.crosstab(df.FamilySize, df.Survived)
# print(df_family)
survived = list(df_family[1])
deaths = list(df_family[0])
w = 0.30
plt.bar(x - w / 2, survived, label='Survived', color='g', width=w)
plt.bar(x + w / 2, deaths, label='Deaths', color='r', width=w)
plt.xticks(ticks=x, labels=family)
plt.ylabel('Number of people')
plt.title('Survival Family size based')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# cabin or not vs survival graph
plt.style.use('fivethirtyeight')
cabin = ['With no Cabin', 'With Cabin']
x = np.arange(len(cabin))
df_cabin = pd.crosstab(df.Cabin, df.Survived)
# print(df_cabin)
survived = list(df_cabin[1])
deaths = list(df_cabin[0])
w = 0.30
plt.bar(x - w / 2, survived, label='Survived', color='g', width=w)
plt.bar(x + w / 2, deaths, label='Deaths', color='r', width=w)
plt.xticks(ticks=x, labels=cabin)
plt.ylabel('Number of people')
plt.title('Survival Cabin reservation based')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# embarked vs survival graph
plt.style.use('fivethirtyeight')
embarked = ['Southampton', 'Cherbourg', 'Queenstown']
x = np.arange(len(embarked))
df_embarked = pd.crosstab(df.Embarked, df.Survived)
# print(df_embarked)
survived = list(df_embarked[1])
deaths = list(df_embarked[0])
w = 0.30
plt.bar(x - w / 2, survived, label='Survived', color='g', width=w)
plt.bar(x + w / 2, deaths, label='Deaths', color='r', width=w)
plt.xticks(ticks=x, labels=embarked)
plt.ylabel('Number of people')
plt.title('Survival Embarkation based')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
