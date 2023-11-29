import pandas as pd

feature_names = {
    'AGEP': 'Age',
    'COW': 'Class of Worker',
    'SCHL': 'Education',
    'MAR': 'Marital Status',
    'OCCP': 'Occupation',
    'POBP': 'Place of Birth',
    'WKHP': 'Hrs worked per week',
    'SEX': 'Gender',
    'RAC1P': 'Race'
}

pums_df = pd.read_csv('src/raw/PUMS_data_dictionary.csv')


def get_feature_dict(feature_name):
    df = pums_df.loc[(pums_df['RT'] == feature_name) & (pums_df['NAME'] == 'VAL')]
    df['Code'] = df['Code'].where(df['Code'].str.isnumeric(), 0).astype(int)
    return dict(zip(df['Code'], df['Value']))

# categorical features
class_of_worker = get_feature_dict('COW')
education = get_feature_dict('SCHL')
marital_status = get_feature_dict('MAR')
occupation = get_feature_dict('OCCP')
place_of_birth = get_feature_dict('POBP')
gender = get_feature_dict('SEX')
race = get_feature_dict('RAC1P')

categorical_feature_idx = [1, 2, 3, 4, 5, 7, 8]
