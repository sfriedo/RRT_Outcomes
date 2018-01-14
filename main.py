from hanaconnection import HanaConnection
from sklearn.naive_bayes import MultinomialNB
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
from knnimpute import knn_impute_few_observed


ATTRIBUTES = ['GENDER', 'ETHNICITY', 'AGE', 'HEIGHT', 'WEIGHT', 'BMI',
              'DIED_90DAYS', 'LENGTH_OF_STAY_HOURS', 'DOSAGE', 'VENT_BEFORE',
              'VENT_AFTER', 'AKIN', 'ELIXHAUSER_VANWALRAVEN',
              'CONGESTIVE_HEART_FAILURE', 'CARDIAC_ARRHYTHMIAS',
              'VALVULAR_DISEASE', 'PULMONARY_CIRCULATION',
              'PERIPHERAL_VASCULAR',
              'HYPERTENSION', 'PARALYSIS', 'OTHER_NEUROLOGICAL',
              'CHRONIC_PULMONARY', 'DIABETES_UNCOMPLICATED',
              'DIABETES_COMPLICATED', 'HYPOTHYROIDISM', 'RENAL_FAILURE',
              'LIVER_DISEASE', 'PEPTIC_ULCER', 'AIDS', 'LYMPHOMA',
              'METASTATIC_CANCER', 'SOLID_TUMOR', 'RHEUMATOID_ARTHRITIS',
              'COAGULOPATHY', 'OBESITY', 'WEIGHT_LOSS', 'FLUID_ELECTROLYTE',
              'BLOOD_LOSS_ANEMIA', 'DEFICIENCY_ANEMIAS', 'ALCOHOL_ABUSE',
              'DRUG_ABUSE', 'PSYCHOSES', 'DEPRESSION', 'OASIS', 'SOFA',
              'SOFA_RENAL', 'SAPS', 'ANIONGAP', 'ALBUMIN', 'BANDS',
              'BICARBONATE', 'BILIRUBIN', 'CREATININE', 'CHLORIDE', 'GLUCOSE',
              'HEMATOCRIT', 'HEMOGLOBIN', 'LACTATE', 'PLATELET', 'POTASSIUM',
              'PTT', 'INR', 'PT', 'SODIUM', 'BUN', 'WBC', 'CR_24_B',
              'GFR_24_B', 'CR_48_B', 'GFR_48_B', 'CR_72_B', 'GFR_72_B',
              'CR_24_A', 'GFR_24_A', 'CR_48_A', 'GFR_48_A', 'CR_72_A',
              'GFR_72_A']
STRING_ATRRIBUTES = ['GENDER', 'ETHNICITY', 'AKIN']
BOOL_ATTRIBUTES = ['DIED_90DAYS', 'ELIXHAUSER_VANWALRAVEN',
                   'CONGESTIVE_HEART_FAILURE', 'CARDIAC_ARRHYTHMIAS',
                   'VALVULAR_DISEASE', 'PULMONARY_CIRCULATION',
                   'PERIPHERAL_VASCULAR',
                   'HYPERTENSION', 'PARALYSIS', 'OTHER_NEUROLOGICAL',
                   'CHRONIC_PULMONARY', 'DIABETES_UNCOMPLICATED',
                   'DIABETES_COMPLICATED', 'HYPOTHYROIDISM', 'RENAL_FAILURE',
                   'LIVER_DISEASE', 'PEPTIC_ULCER', 'AIDS', 'LYMPHOMA',
                   'METASTATIC_CANCER', 'SOLID_TUMOR', 'RHEUMATOID_ARTHRITIS',
                   'COAGULOPATHY', 'OBESITY', 'WEIGHT_LOSS',
                   'FLUID_ELECTROLYTE', 'BLOOD_LOSS_ANEMIA',
                   'DEFICIENCY_ANEMIAS', 'ALCOHOL_ABUSE', 'DRUG_ABUSE',
                   'PSYCHOSES', 'DEPRESSION']
TARGET_ATTRIBUTE = 'DIED_90DAYS'


def get_data():
    query = '''SELECT {} FROM "MIMICIII"."M_RRT_DATA" '''.format(
        ', '.join(ATTRIBUTES))
    result = None
    with HanaConnection() as conn:
        try:
            conn.execute(query)
            result = pd.DataFrame.from_records(conn.fetchall(),
                                               columns=ATTRIBUTES)
        except Exception as e:
            print(e)
    return result


def process_data(df):
    # Extract target attribute
    target = df[TARGET_ATTRIBUTE]
    df = df.drop(TARGET_ATTRIBUTE, axis=1)
    le = preprocessing.LabelEncoder()
    numeric_cols = list(set(ATTRIBUTES) - set(STRING_ATRRIBUTES) -
                        set(BOOL_ATTRIBUTES))
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Convert strings to categorical numeric values
    for column in STRING_ATRRIBUTES:
        df[column] = df[column].fillna('null')
        le.fit(df[column])
        df[column] = le.transform(df[column])
    # Impute missing values
    df[:] = knn_impute_few_observed(df.as_matrix(),
                                    np.isnan(df.as_matrix()), k=3)
    # Normalize values to [0, 1]
    df[:] = preprocessing.MinMaxScaler().fit_transform(df)
    return df, target


def main():
    df = get_data()
    df, target = process_data(df)
    print(df)
    gnb = MultinomialNB()
    scores = cross_val_score(gnb, df, target, cv=5)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    y_pred = gnb.fit(df, target).predict(df)
    print('Confusion matrix: ', confusion_matrix(target, y_pred))


if __name__ == '__main__':
    main()
