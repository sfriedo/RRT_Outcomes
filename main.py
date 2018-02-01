#!python2
from __future__ import division
from hanaconnection import HanaConnection
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import BayesianRidge
from sklearn import preprocessing
from sklearn.model_selection import cross_val_predict, train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, roc_curve, auc, \
    explained_variance_score, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from knnimpute import knn_impute_few_observed
from os import makedirs
from os.path import exists
import logging
import sys
sys.path.append('../sklearn-expertsys-master')
from RuleListClassifier import RuleListClassifier

DATA_TABLE = '"MIMICIII"."RRT_ADDITIONAL_OUTCOMES"'
ATTRIBUTES = ['GENDER', 'ETHNICITY', 'AGE', 'HEIGHT', 'WEIGHT', 'BMI',
              'DIED_90DAYS', 'DOSAGE', 'AKIN', 'ELIXHAUSER_VANWALRAVEN',
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
              'TEMPC', 'SPO2', 'STAY_DAYS', 'VENT_FREE_DAYS']
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
TARGET_ATTRIBUTES = ['DIED_90DAYS', 'STAY_DAYS_LESS_SEVEN',
                     'VENT_DAYS_LESS_SEVEN']
PLOT_DIR = 'results/'


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
            logging.info(e)
    return result


def process_data(df, target_name, scale=True):
    logging.info('Preparing data.')
    # Extract target_name attribute
    logging.info('Encoding attributes.')
    le = preprocessing.LabelEncoder()
    numeric_cols = list(set(ATTRIBUTES) - set(STRING_ATRRIBUTES))
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
    # Convert strings to categorical numeric values
    for column in STRING_ATRRIBUTES:
        df[column] = df[column].fillna('null')
        le.fit(df[column])
        df[column] = le.transform(df[column])
    df['STAY_DAYS_LESS_SEVEN'] = np.where(df['STAY_DAYS'] < 7, 1, 0)
    df = df.drop('STAY_DAYS', axis=1)
    df['VENT_DAYS_LESS_SEVEN'] = np.where(df['VENT_FREE_DAYS'] < 7, 1, 0)
    df = df.drop('VENT_FREE_DAYS', axis=1)
    # Impute missing values
    logging.info('Imputing missing data.')
    df[:] = knn_impute_few_observed(df.as_matrix(),
                                    np.isnan(df.as_matrix()), k=3)
    # Normalize values to [0, 1]
    target_name = target_name.encode("utf-8")
    target = df[target_name]
    for t in TARGET_ATTRIBUTES:
        df = df.drop(t, axis=1)
    if scale:
        logging.info('Scaling data.')
        df[:] = preprocessing.MinMaxScaler().fit_transform(df)
    return df, target


def eval_clf_model(actual, pred):
    conf_matrix = confusion_matrix(actual, pred).ravel()
    tn, fp, fn, tp = conf_matrix
    logging.info('Confusion matrix: ' + str(conf_matrix))
    logging.info('Recall (Sensitivity): ' + str(tp / (tp + fn)))
    logging.info('Specificity: ' + str(tn / (tn + fp)))
    logging.info('Precision: ' + str(tp / (tp + fp)))


def eval_reg_model(actual, pred):
    logging.info('Explained Variance (R2): ' + str(
        explained_variance_score(actual, pred)))
    logging.info('Mean Squared Error: ' + str(
        mean_squared_error(actual, pred)))
    logging.info('Mean Absolute Error: ' + str(
        mean_absolute_error(actual, pred)))


def most_informative_feature_for_binary_classification(feature_names,
                                                       classifier, n=10):
    class_labels = classifier.classes_
    topn_class1 = sorted(zip(classifier.coef_[0], feature_names))[:n]

    for coef, feat in topn_class1:
        logging.info(class_labels[0], coef, feat)


def plot_roc(y_test, y_score, target_name, clf_name):
    fpr, tpr, _ = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = {:.2f})'
             .format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for {} predicting {}'.format(clf_name, target_name))
    plt.legend(loc='lower right')
    if not exists(PLOT_DIR):
        makedirs(PLOT_DIR)
    plt.savefig(PLOT_DIR + 'ROC_{}_{}'.format(target_name, clf_name))
    plt.clf()


def plot_most_important_reg_features(coefs, target_name, num_feats=5):
    coefs = sorted(coefs, key=lambda x: x[0])
    coefs_top = coefs[:num_feats] + coefs[-num_feats:]
    plt.figure(figsize=(14, 5))
    plt.bar(range(len(coefs_top)), [x[0] for x in coefs_top])
    plt.xticks(range(len(coefs_top)), [x[1] for x in coefs_top], fontsize=8)
    if not exists(PLOT_DIR):
        makedirs(PLOT_DIR)
    plt.savefig(PLOT_DIR + 'coef_{}'.format(target_name))
    plt.clf()


def main():
    logging.basicConfig(filename='logfile2.log', level=logging.INFO)
    df_ori = get_data()

    for target_name in TARGET_ATTRIBUTES:
        logging.info('Training for {}'.format(target_name))

        logging.info('Training bayesian rule lists.')
        df, target = process_data(df_ori, target_name, False)
        logging.info('Training cv.')
        brl = RuleListClassifier(class1label=target_name,
                                 verbose=False)
        y_pred = cross_val_predict(brl, df.as_matrix(), target.as_matrix(),
                                   cv=5)
        logging.info('Accuracy: {}'.format(accuracy_score(target, y_pred)))
        eval_clf_model(target, y_pred)

        logging.info('Training neural network.')
        df, target = process_data(df_ori, target_name, True)

        logging.info('Training cv.')
        clf = MLPClassifier(activation='relu',
                            solver='adam', alpha=1e-5,
                            hidden_layer_sizes=(5, 2), random_state=1)
        y_pred = cross_val_predict(clf, df, target, cv=5)
        logging.info('Accuracy: {}'.format(accuracy_score(target, y_pred)))
        eval_clf_model(target, y_pred)

        logging.info('Training MLP on train/test split.')
        y_pred = [0]
        while len([x for x in y_pred if x == 1.0]) < 1:
            X_train, X_test, \
                y_train, y_test = train_test_split(df, target,
                                                   test_size=.3,
                                                   stratify=target)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
        eval_clf_model(y_test, y_pred)
        y_pred = [x[1] for x in clf.predict_proba(X_test)]
        plot_roc(y_test, y_pred, target_name, 'MLPClassifier')

        logging.info('Training BRL on train/test split.')
        brl.fit(X_train.as_matrix(), y_train.as_matrix(),
                feature_labels=df.columns)
        y_pred = brl.predict(X_test.as_matrix())
        eval_clf_model(y_test, y_pred)
        y_pred = [x[1] for x in brl.predict_proba(X_test.as_matrix())]
        logging.info(str(brl))
        plot_roc(y_test, y_pred, target_name, 'BRL')

        logging.info('Training bayesian ridge regression as mimic model.')
        y_train = [x[1] for x in clf.predict_proba(X_train)]
        bayes = BayesianRidge()
        y_pred = bayes.fit(X_train, y_train).predict(X_test)
        logging.info(zip(bayes.coef_, df.columns))
        eval_reg_model(y_test, y_pred)
        plot_most_important_reg_features(zip(bayes.coef_, df.columns),
                                         target_name)
        plot_roc(y_test, y_pred, target_name, 'BayesianRidge')


if __name__ == '__main__':
    main()
