import pickle

import pandas as pd

df = pd.read_csv('adultasd_dataset.csv', na_values='?')

df.drop(df.loc[:, 'A1_Score':'A10_Score'].columns, axis=1, inplace=True)
df.rename(columns={'result numeric': 'A_Score_Results'}, inplace=True)
df.rename(columns={'Class/ASD': 'Class'}, inplace=True)

thresh = df[df['age'] > 51].index
df.drop(thresh, inplace=True)

df['age'] = df['age'].fillna(df['age'].mean())
df['age'].isnull().sum()

df['ethnicity'] = df['ethnicity'].fillna('Unknown')
df['relation'] = df['relation'].fillna('Unknown')

df['ethnicity'] = df['ethnicity'].map(lambda x: x.strip('\''))
df['relation'] = df['relation'].map(lambda x: x.strip('\''))
# df['country_of_residence'] = df['country_of_residence'].map(lambda x: x.strip('\''))
df['age_desc'] = df['age_desc'].map(lambda x: x.strip('\''))
df.columns = df.columns.str.strip()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

from sklearn.utils import resample

df_majority = df[df.Class == 0]
df_minority = df[df.Class == 1]

# Upsample minority class
# random_state used for reproducible results
df_minority_upsampled = resample(df_minority, replace=True, n_samples=450, random_state=23)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Display new class counts
# df_upsampled.Class.value_counts()

X = df_upsampled.drop(columns='Class')
y = df_upsampled['Class']

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

from sklearn.preprocessing import StandardScaler

# scaler = StandardScaler()
# rescaledX_train = scaler.fit_transform(X_train)
# rescaledX_test = scaler.fit_transform(X_test)

# from sklearn.ensemble import AdaBoostClassifier
# model = AdaBoostClassifier(random_state=10)
# a=model.fit(X_train, y_train)
# abc_predict = model.predict(X_test)
# model.score(X_test,y_test)

from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB

# Define Gaussian NB model from sklearn library
nb = GaussianNB()
a = nb.fit(X_train, y_train)

pickle.dump(a, open("asd.pkl", 'wb'))
model = pickle.load(open('asd.pkl', 'rb'))
print(model.predict([[18.0, 1, 4, 0, 1, 6, 0, 2]]))
print(model.predict([[40.0, 0, 9, 0, 0, 2, 0, 5]]))
print(model.predict([[33.0, 1, 10, 0, 0, 10, 0, 3]]))
