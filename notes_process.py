import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re


def get_column_value(df, column_name: str):
    return df[column_name]


def dropNA(arr):
    new = []
    for item in arr:
        if item != '':
            new.append(item)
    return new


def main():
    dataset = pd.read_csv('data/dataset.csv')
    notes = np.array(get_column_value(dataset, 'Notes'))
    notes = dropNA(notes)
    for note in notes:
        note = re.sub("[^a-zA-Z0-9]", " ", note)

    lab_status = get_column_value(dataset, 'Lab Status')  # label
    for item in lab_status:
        if item == 'Negative ID':
            item = 0
        elif item == 'Positive ID':
            item = 1

    corpus = notes
    label = lab_status
    vectorizer = TfidfVectorizer()
    result = vectorizer.fit_transform(corpus).toarray()
    # print(vectorizer.get_feature_names())
    print(result.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        result, label, test_size=0.33, random_state=42)

    # Training the classifier & predicting on test data
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    report = classification_report(y_test, y_pred)
    print('\n Accuracy: ', accuracy_score(y_test, y_pred))
    print('\nClassification Report')
    print('======================================================')
    print('\n', report)


if __name__ == "__main__":
    main()
