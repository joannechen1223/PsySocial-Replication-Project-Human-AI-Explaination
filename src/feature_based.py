import random
import pandas as pd
import pickle

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Lime
import lime
import lime.lime_tabular

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


explainer = lime.lime_tabular.LimeTabularExplainer(
    features[:train.shape[0]].to_numpy(),
    class_names=['<= $50k', '> $50k'],
    feature_names=feature_names,
    categorical_features=categorical_feature_idx,
    categorical_names=categorical_names,
)


ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), categorical_feature_idx)], remainder='passthrough')
ct = ct.fit(features)


instances = [random.randint(0, features_test.shape[0]) for _ in range(10)]
print(instances)
predict_fn = lambda x: rf.predict_proba(ct.transform(x))
for i in range(10):
    exp = explainer.explain_instance(
        features[train.shape[0]:].iloc[instances[i]].to_numpy(),
        predict_fn,
        num_features=10
    )
    exp.save_to_file('assets/feature_based/lime_%s.html' % instances[i])
