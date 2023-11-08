# Data Source
from folktables import ACSDataSource, ACSIncome

# Data Preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split

# Model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Data Source
data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
data = data_source.get_data(states=['CA'], download=True)
features, labels, _ = ACSIncome.df_to_pandas(data)

# Data Preprocessing
le = LabelEncoder()
categorical_features = [1, 2, 3, 4, 5, 6, 8, 9]
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), categorical_features)], remainder='passthrough')

X = ct.fit_transform(features)
y = le.fit_transform(labels)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Testing
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
