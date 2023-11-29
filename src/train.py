import os
import pandas as pd
import pickle

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from raw.features_dict import feature_names, categorical_feature_idx

# Read data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
features_train = train.iloc[:, :-1]
label_train = train.iloc[:, -1]
features_test = test.iloc[:, :-1]
label_test = test.iloc[:, -1]

features = pd.concat([features_train, features_test])
label = pd.concat([label_train, label_test])


# Data Preprocessing
le = LabelEncoder()

feature_names = list(feature_names.values())
categorical_names = {}
for idx in categorical_feature_idx:
    col = feature_names[idx]
    le.fit(features[col])
    features[col] = le.transform(features[col])
    categorical_names[col] = le.classes_

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), categorical_feature_idx)], remainder='passthrough')
ct = ct.fit(features)


X = ct.fit_transform(features)
y = le.fit_transform(label)

X_train = X[:features_train.shape[0]]
X_test = X[features_train.shape[0]:]
y_train = y[:features_train.shape[0]]
y_test = y[features_train.shape[0]:]

# Training
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(X_train.shape)

# Testing
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Export to pickle
pickle.dump(rf, open('models/rf.pkl', 'wb'))

# Export to CSV
dir = './data'
if not os.path.exists(dir):
    os.mkdir(dir)

pd_pred = pd.DataFrame(y_pred, columns=['Income above 50k'])
pd_pred['Income above 50k'] = pd_pred['Income above 50k'].map({0: 'False', 1: 'True'})
pd_pred.to_csv('data/pred.csv', index=False)
