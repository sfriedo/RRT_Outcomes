#!python2
from __future__ import division
from hanaconnection import HanaConnection
from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import confusion_matrix, explained_variance_score, \
    mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from knnimpute import knn_impute_few_observed
import sys
sys.path.append('../sklearn-expertsys-master')
from RuleListClassifier import RuleListClassifier

DATA_TABLE = '"MIMICIII"."M_RRT_DATA_VITALS"'
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
              'GFR_72_A', 'HEARTRATE', 'SYSBP', 'DIASBP', 'MEANBP', 'RESPRATE',
              'TEMPC', 'SPO2']
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
TARGET_ATTRIBUTES = ['DIED_90DAYS', 'STAY_DAYS', 'VENT_FREE_DAYS']


def get_data():
    query = '''SELECT {} FROM {} '''.format(
        ', '.join(ATTRIBUTES), DATA_TABLE)
    result = None
    with HanaConnection() as conn:
        try:
            conn.execute(query)
            result = pd.DataFrame.from_records(conn.fetchall(),
                                               columns=ATTRIBUTES)
        except Exception as e:
            print(e)
    return result


def process_data(df, target, scale=True):
    print('Preparing data.')
    # Extract target attribute
    print('Encoding attributes.')
    le = preprocessing.LabelEncoder()
    numeric_cols = list(set(ATTRIBUTES) - set(STRING_ATRRIBUTES))
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Convert strings to categorical numeric values
    for column in STRING_ATRRIBUTES:
        df[column] = df[column].fillna('null')
        le.fit(df[column])
        df[column] = le.transform(df[column])
    # Impute missing values
    print('Imputing missing data.')
    df[:] = knn_impute_few_observed(df.as_matrix(),
                                    np.isnan(df.as_matrix()), k=3)
    # Normalize values to [0, 1]
    target = df[target]
    df = df.drop(target, axis=1)
    if scale:
        print('Scaling data.')
        df[:] = preprocessing.MinMaxScaler().fit_transform(df)
    return df, target


def eval_clf_model(actual, pred):
    conf_matrix = confusion_matrix(actual, pred).ravel()
    tn, fp, fn, tp = conf_matrix
    print('Confusion matrix: ', conf_matrix)
    print('Recall (Sensitivity): ', tp / (tp + fn))
    print('Specificity: ', tn / (tn + fp))
    print('Precision: ', tp / (tp + fp))


def eval_reg_model(actual, pred):
    print('Explained Variance (R2): ', explained_variance_score(actual, pred))
    print('Mean Squared Error: ', mean_squared_error(actual, pred))
    print('Mean Absolute Error: ', mean_absolute_error(actual, pred))


def most_informative_feature_for_binary_classification(feature_names,
                                                       classifier, n=10):
    class_labels = classifier.classes_
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]

    for coef, feat in topn_class1:
        print(class_labels[0], coef, feat)


def main():
    df = get_data()

    for target in TARGET_ATTRIBUTES:
        df, target = process_data(df, target)
        X_train, X_test, y_train, y_test = train_test_split(
            df, target, test_size=0.3)

        print('Training bayesian rule lists.')
        brl = RuleListClassifier(max_iter=10000, class1label=target,
                                 verbose=False)
        scores = cross_val_score(brl, df.as_matrix(), target.as_matrix(), cv=5)
        print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),
              scores.std() * 2))
        y_pred = brl.fit(X_train.as_matrix(), y_train.as_matrix(),
                         feature_labels=df.columns) \
            .predict(X_test)
        print(brl)
        eval_clf_model(y_test, y_pred)

        print('Training neural network.')
        clf = MLPClassifier(activation='relu',
                            solver='adam', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
        scores = cross_val_score(clf, df, target, cv=5)
        print('Accuracy: %0.2f (+/- %0.2f)' % (scores.mean(),
              scores.std() * 2))
        y_pred = clf.fit(X_train, y_train).predict(X_test)
        eval_clf_model(y_test, y_pred)

        print('Training bayesian rule lists as mimic model.')
        y_train = [x[1] for x in clf.predict_proba(X_train)]
        y_pred = brl.fit(X_train, y_train).predict(X_test)
        eval_clf_model(y_test, y_pred)

if __name__ == '__main__':
    main()
