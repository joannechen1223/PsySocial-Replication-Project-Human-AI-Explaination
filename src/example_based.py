import os
import pandas as pd
import pickle

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

# Lime

from raw.features_dict import feature_names, categorical_feature_idx

# Load random forest model
rf = None
with open('models/rf.pkl', 'rb') as f:
    rf = pickle.load(f)

# Read data
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
features_train = train.iloc[:, :-1]
label_train = train.iloc[:, -1]
features_test = test.iloc[:, :-1]
label_test = test.iloc[:, -1]

# example_based = pd.read_csv('data/example_based_selected.csv')
# features_example_based = example_based.iloc[:, :-1]
# label_example_based = example_based.iloc[:, -1]

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

# Fit example_based data
X = ct.fit_transform(features)
y = le.fit_transform(label)

X_train = X[:features_train.shape[0]]
X_test = X[features_train.shape[0]:]
y_train = y[:features_train.shape[0]]
y_test = y[features_train.shape[0]:]

y_train_pred = rf.predict(X_train)
accuracy = accuracy_score(y_train, y_train_pred)
print("Accuracy:", accuracy)


# Concatenate example_based data and predict result
pd_pred = pd.DataFrame(y_train_pred, columns=['Income above 50k(pred)'])
pd_pred['Income above 50k(pred)'] = pd_pred['Income above 50k(pred)'].map({0: 'False', 1: 'True'})
print(pd_pred.head())
train_pred = pd.concat([train, pd_pred], axis=1)

# Export to CSV
dir = './data'
if not os.path.exists(dir):
    os.mkdir(dir)

train_pred.to_csv('data/train_pred.csv', index=False)
