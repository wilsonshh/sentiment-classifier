from sklearn import svm
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from data_preparation import data_preparation


def run_svm():
    """
    perform linear support vector machine algorithm on the extracted data values

    Output:
            cross validation score for linear support vector machine
    """
    # Assigning data preparation function to the values
    values = data_preparation()

    # Assigning train_data that is returned from data preparation  class
    train_data = values[0]

    # Assigning train_vector that is returned from data preparation class
    train_vector = values[1]

    support_vector_classification = svm.LinearSVC()
    print "Training Support Vector Machine"
    support_vector_classification = GridSearchCV(support_vector_classification, param_grid={'C': [1, 10, 100, 1000]}, cv=(StratifiedKFold(n_splits=10, random_state=None)))
    support_vector_classification = support_vector_classification.fit(train_vector, train_data.sentiment)
    print "Cross validation score:", support_vector_classification.best_score_