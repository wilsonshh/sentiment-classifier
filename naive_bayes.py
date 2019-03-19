from sklearn import naive_bayes
from sklearn.model_selection import cross_val_score
from data_preparation import data_preparation

def run_mnb():
    """
    perform multinomial naive bayes on the extracted data values

    Output:
            cross validation score for multinomial naive bayes classifier
    """
    values = data_preparation()
    train_vector = values[1]
    train_data = values[0]
    nb = naive_bayes.MultinomialNB()
    cv_score = cross_val_score(nb, train_vector, train_data.sentiment, cv=10)
    print "Training Naive Bayes"
    print "Cross Validation Score = ", cv_score.mean()