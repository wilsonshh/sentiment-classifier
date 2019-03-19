import pandas as pd
from data_parser import clean_text
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection.univariate_selection import chi2, SelectKBest

# The number of dimensions for chi2 and truncated svd function
number_of_dimensions = 800

def data_preparation():
    """
    performs data preparation in multiple stages involving cleaning, feature extraction,
    dimension reduction and feature scaling

    Returns:
            train_data, train_vector, test_vector
    """
    train_list = []
    test_list = []

    train_data = pd.read_csv("train.csv", header=0)
    test_data = pd.read_csv("test.csv", header=0)

    for i in xrange(0, len(train_data.review)):
        # Append raw review texts as vector
        train_list.append(clean_text(train_data.review[i]))

    for i in xrange(0, len(test_data.review)):
        # Append raw review texts as vector
        test_list.append(clean_text(test_data.review[i]))

    # Create vectors from words
    count_vec = TfidfVectorizer(analyzer="word", max_features=15000, ngram_range=(1, 2))
    train_vector = count_vec.fit_transform(train_list)
    test_vector = count_vec.transform(test_list)

    # Dimension Reduction based on chi2, this works with both
    print "Processing chi2 dimension reduction"
    fselect = SelectKBest(chi2, k=number_of_dimensions)
    train_vector = fselect.fit_transform(train_vector, train_data.sentiment)
    test_vector = fselect.transform(test_vector)

    # Convert into numpy arrays
    train_vector = train_vector.toarray()
    test_vector = test_vector.toarray()

    # Feature Scaling: support vector machine requires scaling and works with any
    # scaling technique we outline here however, naive bayes only works with min-max
    # scaler with the feature range 0, 1

    print "Scaling vectors"
    # scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    # scaler = preprocessing.StandardScaler()
    scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))
    train_vector = scaler.fit_transform(train_vector)
    test_vector = scaler.fit_transform(test_vector)

    return train_data, train_vector, test_data, train_data
