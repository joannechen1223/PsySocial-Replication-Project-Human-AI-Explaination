import random

# Data Source
from folktables import ACSDataSource, ACSIncome

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Lime
import lime
import lime.lime_tabular

# Data Source
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=['NY'], download=True)
features, labels, _ = ACSIncome.df_to_pandas(data)

print(ACSIncome.features)

# Data Preprocessing
le = LabelEncoder()
feature_names = ['Age', 'Class of Worker', 'Education', 'Marital Status', 'Occupation',
                 'Place of Birth', 'Relationship', 'Hrs worked per week', 'Sex', 'Race']
categorical_features = [1, 2, 3, 4, 5, 6, 8, 9]
categorical_names = {}
for idx in categorical_features:
    col = ACSIncome.features[idx]
    le.fit(features[col])
    features[col] = le.transform(features[col])
    categorical_names[col] = le.classes_

# print(categorical_names)
# print(features[801:1000].shape)

ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')
ct = ct.fit(features)


X = ct.fit_transform(features)
y = le.fit_transform(labels)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Training
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
print(X_train.shape)

# Testing
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

n_train = X_train.shape[0]
n_test = X_test.shape[0]
features_train = features[:n_train]
features_test = features[n_train:]

# print(type(X_train))
explainer = lime.lime_tabular.LimeTabularExplainer(
    features_train.to_numpy(),
    class_names=['<= $50k', '> $50k'],
    feature_names=feature_names,
    categorical_features=categorical_features,
    categorical_names=categorical_names,
)

print(features_test.shape)
instances = [random.randint(0, n_test) for _ in range(10)]
predict_fn = lambda x: rf.predict_proba(ct.transform(x))
for i in range(10):
    # print(instances[i], y_pred[instances[i]], y_test[instances[i]])
    # print(features_test.iloc[instances[i]])
    exp = explainer.explain_instance(features_test.iloc[instances[i]].to_numpy(), predict_fn, num_features=10)
    exp.save_to_file('assets/feature_based/lime_%s.html' % instances[i])
