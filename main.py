import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

def main():
    # transfère du doc csv
    df = pd.read_csv('language-identification-datasets.csv')

    # division en 2 array pour text et langue
    X = df['Text']
    y = df['Language']
    z = df['ID']
    X_train = X[0:24326]
    X_test = X[24300:31014]
    y_train = y[0:24326]
    y_test = y[24300:31014]
    z_test = z[24300:31014]

    # creation model
    model = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('classifier', DecisionTreeClassifier())
    ])

    # apprentissage
    model.fit(X_train, y_train)

    # évalutation model
    y_pred = model.predict(X_test)
    pd.DataFrame(y_pred, z_test).to_csv('ModèleFinale.csv')

if __name__ == "__main__":
  main()