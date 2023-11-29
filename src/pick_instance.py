import os
import pandas as pd

from sklearn.metrics import accuracy_score

# Read data
test = pd.read_csv('data/test.csv')
pred = pd.read_csv('data/pred.csv')
label_test = test.iloc[:, -1]

pred.rename(columns={'Income above 50k': 'Income above 50k(pred)'}, inplace=True)

# Accuracy
accuracy = accuracy_score(label_test, pred)
print("Accuracy:", accuracy)

pd_test = pd.concat([test, pred], axis=1)

# Separate predict correct and incorrect data
pd_test_correct = pd_test[pd_test['Income above 50k'] == pd_test['Income above 50k(pred)']]
pd_test_incorrect = pd_test[pd_test['Income above 50k'] != pd_test['Income above 50k(pred)']]

# Randomly pick 9 correct and 6 incorrect data
correct_sample = pd_test_correct.sample(n=9, random_state=1)
incorrect_sample = pd_test_incorrect.sample(n=6, random_state=1)

correct_sample_idx = correct_sample.index
incorrect_sample_idx = incorrect_sample.index

# Export to CSV
dir = './data'
if not os.path.exists(dir):
    os.mkdir(dir)

correct_sample.to_csv('data/correct_sample.csv', index=False)
incorrect_sample.to_csv('data/incorrect_sample.csv', index=False)

print(correct_sample)
print(incorrect_sample)
