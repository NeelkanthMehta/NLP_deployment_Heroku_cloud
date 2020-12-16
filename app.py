import pickle
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from flask import Flask, render_template, url_for, request


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        with open('nlp_model.pkl', 'rb') as modelfile:
            classifier = pickle.load(modelfile)

        with open('transform.pkl', 'rb') as transformfile:
            cv = pickle.load(transformfile)

        message = request.form['message']
        vectorized = cv.transform([message]).toarray()
        y_predict = classifier.predict(vectorized)

    return render_template('result.html', prediction=y_predict)

if __name__ == '__main__':
    app.run(debug=True)
