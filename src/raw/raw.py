import os
import pandas as pd

# Data Source
from folktables import ACSDataSource, ACSIncome

from sklearn.model_selection import train_test_split

import features_dict

# Data Source
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=['NY'], download=True)
features, labels, _ = ACSIncome.df_to_pandas(data)

print(ACSIncome.features)

features = features[features_dict.feature_names.keys()]


features['COW'] = features['COW'].map(features_dict.class_of_worker)
features['SCHL'] = features['SCHL'].map(features_dict.education)
features['MAR'] = features['MAR'].map(features_dict.marital_status)
features['OCCP'] = features['OCCP'].map(features_dict.occupation)
features['POBP'] = features['POBP'].map(features_dict.place_of_birth)
features['SEX'] = features['SEX'].map(features_dict.gender)
features['RAC1P'] = features['RAC1P'].map(features_dict.race)

# Rename columns
features.rename(columns=features_dict.feature_names, inplace=True)
labels.rename(columns={'PINCP': 'Income above 50k'}, inplace=True)

print(features.head())
print(labels.head())

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=True)

# concatenate X_train and y_train
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Export to CSV
dir = './data'
if not os.path.exists(dir):
    os.mkdir(dir)

train.to_csv('data/train.csv', index=False)
test.to_csv('data/test.csv', index=False)
