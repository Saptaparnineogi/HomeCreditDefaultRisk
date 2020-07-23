import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score


def read_csv(file):
    """
    :param file: csv file to read
    :return: created dataframe
    """
    df = pd.read_csv(file)
    return df


def match_dtypes(train, test, target_name='TARGET'):
    """
    This function converts test dataframe to match columns in accordance with the
    training dataframe.
    """
    for column_name in train.drop([target_name], axis=1).columns:
        test[column_name] = test[column_name].astype(train[column_name].dtype)
    return test


def column_encoding(train, test):
    """
    Create a label encode object having less than or equal
    to 2 unique values.
    Create one hot encoding for categorical variables.

    :param train: training data
    :param test: test data
    :return: train, test
    """
    le = LabelEncoder()
    transform_counter = 0
    # iterate through all the categorical columns
    for col in train.select_dtypes('object').columns:
        # select only those columns where number of unique values in the category is less than or equal to 2
        if pd.Series.nunique(train[col]) <= 2:
            train[col] = le.fit_transform(train[col].astype(str))
            test[col] = le.fit_transform(test[col].astype(str))
            transform_counter += 1

    # one-hot encode of categorical variables
    train = pd.get_dummies(train, drop_first=True)
    test = pd.get_dummies(test, drop_first=True)
    target = train['TARGET']
    train, test = train.align(test, axis=1, join='inner')
    train['TARGET'] = target
    return train, test


def check_anomalies(train, test):
    """
    We have redundant values in the column 'YEARS_EMPLOYED'.
    This function checks for anomalies in data and replace
    those with null values.

    :param train: training data
    :param test: test data
    :return: train, test
    """
    # Create an anomalous flag column
    train['DAYS_EMPLOYED_ANOM'] = train["DAYS_EMPLOYED"] == 365243

    # Replace the anomalous values with nan
    train['DAYS_EMPLOYED'] = train['DAYS_EMPLOYED'].replace({365243: np.nan})
    # Create an anomalous flag column in test set
    test['DAYS_EMPLOYED_ANOM'] = test["DAYS_EMPLOYED"] == 365243
    # Replace the anomalous values with nan in test set
    test['DAYS_EMPLOYED'] = test['DAYS_EMPLOYED'].replace({365243: np.nan})
    return train, test


def high_correlation(train):
    """
    This function checks the correlation of features with the TARGET variable
    :param train: training dataframe
    :return: list of features with higher correlation
    """
    corr_df = train.corr()["TARGET"].sort_values().to_frame()
    corr_df['TARGET'] = corr_df['TARGET'].astype(float)
    corr_df = corr_df[(corr_df['TARGET'] > 0.07) & (corr_df['TARGET'] < 1) | (corr_df['TARGET'] < -0.15)]
    corr_df = corr_df.reset_index()
    corr_features = corr_df['index'].tolist()
    return corr_features


def feature_engg(train, test):
    """
    In this function we perform some feature engineering,
    Like Imputation, Polynomial feature creation,
    and add few more calculated features

    :param train: training dataframe
    :param test: test dataframe
    :return: train and test dataframe after feature engineering
    """
    fitting_vars = high_correlation(train)
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    train[fitting_vars] = imputer.fit_transform(train[fitting_vars])
    test[fitting_vars] = imputer.transform(test[fitting_vars])
    poly_feat = PolynomialFeatures(degree=4)
    poly_train_df = poly_feat.fit_transform(train[fitting_vars])
    poly_test_df = poly_feat.fit_transform(test[fitting_vars])
    # build dataframe with Polynomial interaction variables
    poly_train_df = pd.DataFrame(poly_train_df, columns=poly_feat.get_feature_names(fitting_vars))
    poly_test_df = pd.DataFrame(poly_test_df, columns=poly_feat.get_feature_names(fitting_vars))
    # Add target to the newly created dataframe so that we can determine correlation
    poly_train_df['TARGET'] = train['TARGET']
    # finding put correlation between newly created variables and TARGET
    interaction = poly_train_df.corr()['TARGET'].sort_values()

    # Choose the columns having highest correlation to the Target variable
    selected_inter_variables = list(
        set(interaction.head(15).index).union(interaction.tail(15).index).difference(set({'1', 'TARGET'})))
    # Get a list of unselected variables which can be dropped
    unselected_cols = [element for element in poly_train_df.columns if element not in selected_inter_variables]
    poly_train_df = poly_train_df.drop(unselected_cols, axis=1)
    poly_test_df = poly_test_df.drop(list(set(unselected_cols).difference({'TARGET'})), axis=1)
    train = train.join(poly_train_df.drop(['EXT_SOURCE_2', 'EXT_SOURCE_3'], axis=1))
    test = test.join(poly_test_df.drop(['EXT_SOURCE_2', 'EXT_SOURCE_3'], axis=1))

    # We add some more calculated features which were proven to be important
    # based on an article published by Wells Fargo
    train['DIR'] = train['AMT_CREDIT'] / train['AMT_INCOME_TOTAL']
    train['AIR'] = train['AMT_ANNUITY'] / train['AMT_INCOME_TOTAL']
    train['ACR'] = train['AMT_ANNUITY'] / train['AMT_CREDIT']
    train['DAR'] = train['DAYS_EMPLOYED'] / train['DAYS_BIRTH']
    test['DIR'] = test['AMT_CREDIT'] / test['AMT_INCOME_TOTAL']
    test['AIR'] = test['AMT_ANNUITY'] / test['AMT_INCOME_TOTAL']
    test['ACR'] = test['AMT_ANNUITY'] / test['AMT_CREDIT']
    test['DAR'] = test['DAYS_EMPLOYED'] / test['DAYS_BIRTH']
    return train, test


def prepare_data_for_training(train, test):
    """
    This function performs feature imputation for all columns
    with the median values.
    This also does the normalization of features.
    """
    # Feature Imputation
    features = list(set(train.columns).difference({'TARGET'}))
    imputer = SimpleImputer(strategy="median")
    # Feature scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    imputer.fit(train.drop(['TARGET'], axis=1))
    train_transformed = imputer.transform(train.drop(['TARGET'], axis=1))
    test_transformed = imputer.transform(test)
    train_transformed = scaler.fit_transform(train_transformed)
    test_transformed = scaler.transform(test_transformed)
    return train_transformed, test_transformed


def model_training(train, test, target, test_not_transformed):
    """
    We are using Logistic regression to predict our target variable

    :param train: training dataframe after all preprocessing
    :param test: test dataframe after all preprocessing
    :param target: our target variable to predict
    :param test_not_transformed: test dataframe without preprocessing
    """
    x_training_set, x_validation_set, y_training_set, y_validation_set = train_test_split(train,
                                                                                          target, test_size=0.33,
                                                                                          random_state=42)
    logistic_regressor = LogisticRegression(C=2)
    logistic_regressor.fit(x_training_set, y_training_set)
    log_regression_pred = logistic_regressor.predict(x_validation_set)
    logistic_test_pred = logistic_regressor.predict(test)
    pd.DataFrame({'target': logistic_test_pred})['target'].value_counts()

    print("The accuracy is: ", accuracy_score(y_validation_set, log_regression_pred))
    print("\n")
    print("The classification report is:\n", classification_report(y_validation_set, log_regression_pred))
    print("ROC AUC score is: ", roc_auc_score(y_validation_set, log_regression_pred))

    # We want to predict the probability for not repaying the loan.
    # So we would use predict.proba method from Logistic regression

    log_regression_pred_test = logistic_regressor.predict_proba(test)
    submission_log_regression = test_not_transformed[['SK_ID_CURR']]
    submission_log_regression['TARGET'] = log_regression_pred_test[:, 1]
    submission_log_regression.to_csv("log_regression.csv", index=False)


if __name__ == '__main__':
    train_df = pd.read_csv('/dataset/application_train.csv')
    test_df = pd.read_csv('/dataset/application_test.csv')
    test_df = match_dtypes(train_df, test_df)
    train_df, test_df = column_encoding(train_df, test_df)
    train_df, test_df = check_anomalies(train_df, test_df)
    train_df, test_df = feature_engg(train_df, test_df)
    train_trans, test_trans = prepare_data_for_training(train_df, test_df)
    model_training(train_trans, test_trans)
