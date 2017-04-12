import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import bs4

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.extmath import density
from sklearn import metrics
from time import time

def get_text(soup):
    for tag in soup.find_all('strong'):
        tag.decompose()
    return soup.get_text()

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

    def extract_content_from_desc(x):
        soup = bs4.BeautifulSoup(x, 'lxml')
        return get_text(soup)

    df['desc_content'] = df['description'].apply(extract_content_from_desc, 1)

    df_fraud = df[df.fraud == 1]
    # random sample not fraud
    df_not_fraud = df[df.fraud == 0].sample(df_fraud.shape[0])
    # create new balanced class dataframe
    df = pd.concat([df_fraud, df_not_fraud], ignore_index=True)

    df = df[df['sale_duration'] >= 0] # drop outlier case where sale duration is negative

    X = df['desc_content'].values
    y = df['fraud'].values

    # X_train, X_test, y_train, y_test
    # return train_test_split(X, y)
    return X, y

def model(X, y, max_df_=.90, min_df_=.001, ngram=(1,2)):

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    stopwords = set(list(ENGLISH_STOP_WORDS))

    tfidf = TfidfVectorizer(max_features=10000, max_df = max_df_, min_df=min_df_, stop_words = stopwords, ngram_range = ngram)

    tfidf.fit(X_train)
    vector = tfidf.transform(X_train)

    nb = GaussianNB()
    nb.fit(vector.todense(), y_train)

    return nb, tfidf, X_test, y_test

def print_top_words(nb, tfidf, top_n_words=10):
        # printing top words for each emoji
    print ''
    print '----- Top {} words for each label in Train set'.format(top_n_words)
    print '-'*60
    bag = np.array(tfidf.get_feature_names())
    for i in range(len(nb.classes_)):
        top =  bag[nb.theta_[i].argsort()[::-1]][:top_n_words]
        print nb.classes_[i], ' -->', top
    print ''

def test_GaussianNB(X, y):
    nb, tfidf, X_test, y_test = model(X, y)
    test_tfidf = tfidf.transform(X_test)
    predicted = nb.predict(test_tfidf.todense())
    acc = np.mean(y_test == predicted)
    print 'Test accuracy =', acc
    print ''
    print_top_words(nb, tfidf)

def benchmark(clf):

    target_names = ['not_fraud', 'fraud']

    print '_' * 80
    print "Training: "
    print clf
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print "train time: %0.3fs" % train_time

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print "test time:  %0.3fs" % test_time

    score = metrics.accuracy_score(y_test, pred)
    print "accuracy:   %0.3f" % score

    if hasattr(clf, 'coef_'):
        print "dimensionality: %d" % clf.coef_.shape[1]
        print "density: %f" % density(clf.coef_)

        # print "top 10 keywords per class:"
        # for i, label in enumerate(target_names):
        #     top10 = np.argsort(clf.coef_[i])[-10:]
        #     print trim("%s: %s" % (label, " ".join(feature_names[top10])))
        # print ""

    print "classification report:"
    print metrics.classification_report(y_test, pred, target_names=target_names)

    print "confusion matrix:"
    print metrics.confusion_matrix(y_test, pred)

    print ""
    clf_descr = str(clf).split('(')[0]
    return clf_descr, score, train_time, test_time

def test_classifiers(X_train, X_test, y_train, y_test, feature_names):
    results = []
    for clf, name in (
            (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
            (Perceptron(n_iter=50), "Perceptron"),
            (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
            (KNeighborsClassifier(n_neighbors=10), "kNN"),
            (RandomForestClassifier(n_estimators=100), "Random forest")):
        print '=' * 80
        print name
        results.append(benchmark(clf))

    for penalty in ["l2", "l1"]:
        print '=' * 80
        print "%s penalty" % penalty.upper()
        # Train Liblinear model
        results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
                                                dual=False, tol=1e-3)))

        # Train SGD model
        results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                               penalty=penalty)))

    # Train SGD with Elastic Net penalty
    print '=' * 80
    print "Elastic-Net penalty"
    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
                                           penalty="elasticnet")))

    # Train NearestCentroid without threshold
    print '=' * 80
    print "NearestCentroid (aka Rocchio classifier)"
    results.append(benchmark(NearestCentroid()))

    # Train sparse Naive Bayes classifiers
    print '=' * 80
    print "Naive Bayes"
    results.append(benchmark(MultinomialNB(alpha=.01)))
    results.append(benchmark(BernoulliNB(alpha=.01)))

    print '=' * 80
    print "LinearSVC with L1-based feature selection"
    # The smaller C, the stronger the regularization.
    # The more regularization, the more sparsity.
    results.append(benchmark(Pipeline([
      ('feature_selection', LinearSVC(penalty="l1", dual=False, tol=1e-3)),
      ('classification', LinearSVC())
    ])))

    # make some plots

    indices = np.arange(len(results))

    results = [[x[i] for x in results] for i in range(4)]

    clf_names, score, training_time, test_time = results
    training_time = np.array(training_time) / np.max(training_time)
    test_time = np.array(test_time) / np.max(test_time)

    plt.figure(figsize=(12, 8))
    plt.title("Score")
    plt.barh(indices, score, .2, label="score", color='navy')
    plt.barh(indices + .3, training_time, .2, label="training time",
             color='c')
    plt.barh(indices + .6, test_time, .2, label="test time", color='darkorange')
    plt.yticks(())
    plt.legend(loc='best')
    plt.subplots_adjust(left=.25)
    plt.subplots_adjust(top=.95)
    plt.subplots_adjust(bottom=.05)

    for i, c in zip(indices, clf_names):
        plt.text(-.3, i, c)

    plt.show()

def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."

def run_tests(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    stopwords = set(list(ENGLISH_STOP_WORDS))

    tfidf = TfidfVectorizer(max_features=10000, max_df=.9, min_df=.001, stop_words=stopwords, ngram_range=(1,2))

    tfidf.fit(X_train)
    vector = tfidf.transform(X_train)

    test_tfidf = tfidf.transform(X_test)

    feature_names = tfidf.get_feature_names()

    print "Extracting 10 best features by a chi-squared test"
    print ""

    ch2 = SelectKBest(chi2, k=10)
    X_train = ch2.fit_transform(vector.todense(), y_train)
    X_test = ch2.transform(test_tfidf.todense())

    feature_names = [feature_names[i] for i in ch2.get_support(indices=True)]

    print "done"
    print ""

    feature_names = np.asarray(feature_names)

    test_classifiers(X_train, X_test, y_train, y_test, feature_names)

if __name__ == '__main__':
    X, y = load_data()
    # run_tests(X, y)
    test_GaussianNB(X, y)
