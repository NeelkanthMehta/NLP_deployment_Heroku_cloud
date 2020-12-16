import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_data(filepath='data/spam.csv'):
    """
    :param filepath: path to data
    :return: text column and label column
    """
    df = pd.read_csv(filepath_or_buffer=filepath, encoding='latin-1')
    df.drop(labels=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1, inplace=True)
    df.columns = ['label', 'message']
    return df['message'], df['label']


def train_model(x, y, verbose=True):
    """
    Fits and trains MultinomialNB model

    param x: explanatory variables
    param y: target variable
    return: test Accuracy score
    """

    # Create Countvectorizer
    cv = CountVectorizer()
    X = cv.fit_transform(x)

    # Save transformers for later use
    with open('transform.pkl', 'wb') as binary_file:
        pickle.dump(cv, binary_file)
    if verbose:
        print('saved CountVectorizer transform variable to transform.pkl file.')

    # Splitting the dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

    # Fit a classifier
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)
    classifier.score(X_test, y_test)

    with open('nlp_model.pkl', 'wb') as binary_file:
        pickle.dump(classifier, binary_file)
    if verbose:
        print('saved MultinomialNB model to nlp_model.pkl')

    # print(classifier.predict(X_test[10]))
    # print(classifier.score(X_test, y_test))


if __name__ == '__main__':
    messages, labels = load_data()
    train_model(x=messages, y=labels)
