# create a model that can detect the language of a text based on the language-identification dataset from Kaggle
# usage: python3 main.py [path_to_data]

import sys
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

def get_data(path: str):
    """Get data from a csv file
    Args:
        path (str): path to the csv file
    Returns:
        pd.DataFrame: dataframe containing the data
    """
    df = pd.read_csv(path, index_col=0)
    return df

def main():
    """Main function
    """
    # get the data
    path = sys.argv[1]
    df = get_data(path)

    # split the data
    X = df['text']
    y = df['language']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # create the model
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', DecisionTreeClassifier())
    ])

    # train the model
    model.fit(X_train, y_train)

    # evaluate the model
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f'Accuracy: {acc}')

if __name__ == "__main__":
    main()

