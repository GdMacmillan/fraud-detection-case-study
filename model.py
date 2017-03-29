import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from collections import OrderedDict
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, mean_squared_error
import datetime

def get_dumm(df, cols):
    all_cols = df.columns
    df = pd.get_dummies(df, columns=cols)

    def diff(first, second):
        second = set(second)
        return [item for item in first if item not in second]

    train_cols = diff(df.columns, all_cols)
    return df, train_cols

def load_data():
    df = pd.read_json('data/data.json')

    train_cols = ['body_length', 'gts', 'num_payouts', 'sale_duration', 'days_user_event', 'name_length', 'has_header', 'has_header_nan', 'has_logo', 'listed', 'user_type']

    df['fraud'] = 0
    def is_fraud(x):
        if x == 'fraudster' or x == 'fraudster_event':
            return 1
        else:
            return 0

    df['fraud'] = df['acct_type'].apply(is_fraud, 1)

    def is_listed(x):
        if x == 'y':
            return 1
        else:
            return 0

    df['listed'] = df['listed'].apply(is_listed, 1)

    df = df[df['sale_duration'] >= 0] # drop outlier case where sale duration is negative

    def is_nan(x):
        if np.isnan(x):
            return 1
        else:
            return 0

    df['has_header_nan'] = df['has_header'].apply(is_nan, 1)
    df['has_header'].fillna(0, inplace=True)

    # create column for length of ticket types list
    df['len_ticket_types'] = df['ticket_types'].apply(lambda x: len(x), 1)

    df['payout_type'] = df['payout_type'].apply(lambda x: 'not_specified' if len(x) ==  0 else x)

    get_dummy_cols = ['channels', 'currency', 'payout_type']

    # create dummy columns for categorical variables
    df, additional_train_cols = get_dumm(df, get_dummy_cols)

    # all fraud cases
    df_fraud = df[df.fraud == 1]
    # random sample not fraud
    df_not_fraud = df[df.fraud == 0].sample(df_fraud.shape[0])
    # create new balanced class dataframe
    df = pd.concat([df_fraud, df_not_fraud], ignore_index=True)

    df = df[df['sale_duration'] >= 0] # drop outlier case where sale duration is negative
    def days_between_timestamp(x):
        return abs(datetime.datetime.fromtimestamp(x[0]) - (datetime.datetime.fromtimestamp(x[1]))).days

    df['days_user_event'] = df[['event_created', 'user_created']].apply(days_between_timestamp, 1)

    train_cols += additional_train_cols
    X = df[train_cols].values
    y = df['fraud'].values

    # X_train, X_test, y_train, y_test
    # return train_test_split(X, y)
    return X, y, train_cols

def validation(model, X, y, n_splits = 5):
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    kf = KFold(n_splits = 5)
    accuracies = []
    precisions = []
    recalls = []
    for train_index, test_index in kf.split(X_train,y_train):
        X_train_fold = X_train[train_index]
        y_train_fold = y_train[train_index]
        X_valid_fold = X_train[test_index]
        y_valid_fold = y_train[test_index]
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_valid_fold)
        accuracies.append(accuracy_score(y_valid_fold, predictions))
        precisions.append(precision_score(y_valid_fold, predictions))
        recalls.append(recall_score(y_valid_fold, predictions))
    mean_accuracy = np.mean(accuracies)
    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    return mean_accuracy, mean_precision, mean_recall

def display_cross_val_results(acc, prec, rec):
    print "accuracy:", acc
    print "precision:", prec
    print "recall:", rec

def display_rmse_feature_importances(clf, X_train, X_test, y_train, y_test, train_cols, model_type):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print 'RMSE: ', np.sqrt(mean_squared_error(y_test, y_pred))
    indx = np.arange(len(train_cols))
    srt = np.argsort(clf.feature_importances_)[::-1]
    feat_names = [str(s) for s in np.array(train_cols)[srt]]
    plt.bar(indx, clf.feature_importances_[srt])
    plt.xticks(indx, feat_names, rotation=45)
    plt.ylabel("Normalized Importances")
    plt.title("Feature Importances for {}".format(model_type))
    plt.tight_layout()
    plt.show()

def results_for_rf(X, y, train_cols):
    clf = RandomForestClassifier(n_estimators=100, oob_score=True)
    display_cross_val_results(*validation(clf, X, y))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    display_rmse_feature_importances(clf, X_train, X_test, y_train, y_test, train_cols, 'RandomForest Classifier')
    clf.fit(X, y)
    print 'OOB_error: ', 1 - clf.oob_score_

def results_for_adaboost(X, y, train_cols):
    clf = AdaBoostClassifier(n_estimators = 100)
    display_cross_val_results(*validation(clf, X, y))
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    display_rmse_feature_importances(clf, X_train, X_test, y_train, y_test, train_cols, 'AdaBoost Classifier')

def plot_error_rate_of_different_ensemble_clfs(X, y):
    ensemble_clfs = [
        ("RandomForestClassifier, max_features='sqrt'",
            RandomForestClassifier(warm_start=True, oob_score=True,
                                   max_features="sqrt")),
        ("RandomForestClassifier, max_features='log2'",
            RandomForestClassifier(warm_start=True, max_features='log2',
                                   oob_score=True)),
        ("RandomForestClassifier, max_features=None",
            RandomForestClassifier(warm_start=True, max_features=None,
                                   oob_score=True))
    ]

    # Map a classifier name to a list of (<n_estimators>, <error rate>) pairs.
    error_rate = OrderedDict((label, []) for label, _ in ensemble_clfs)

    # Range of `n_estimators` values to explore.
    min_estimators = 15
    max_estimators = 175

    for label, clf in ensemble_clfs:
        for i in range(min_estimators, max_estimators + 1):
            clf.set_params(n_estimators=i)
            clf.fit(X, y)

            # Record the OOB error for each `n_estimators=i` setting.
            oob_error = 1 - clf.oob_score_
            error_rate[label].append((i, oob_error))

    # Generate the "OOB error rate" vs. "n_estimators" plot.
    for label, clf_err in error_rate.items():
        xs, ys = zip(*clf_err)
        plt.plot(xs, ys, label=label)

    plt.xlim(min_estimators, max_estimators)
    plt.xlabel("n_estimators")
    plt.ylabel("OOB error rate")
    plt.legend(loc="upper right")
    plt.show()

if __name__ == '__main__':
    X, y, train_cols = load_data()
    # results_for_rf(X, y, train_cols)
    # plot_error_rate_of_different_ensemble_clfs(X, y)
    # results_for_adaboost(X, y, train_cols)
