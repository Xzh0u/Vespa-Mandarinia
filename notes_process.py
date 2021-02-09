import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import copy


def get_column_value(df, column_name: str):
    return df[column_name]


def dropNA(arr):
    new = []
    index = []
    for idx, item in enumerate(arr):
        if item.strip(' ') != '' and len(item.split()) >= 3:
            new.append(item)
            index.append(idx)
    return new, index


def main():
    dataset = pd.read_csv('data/dataset.csv')
    notes = np.array(get_column_value(dataset, 'Notes'))
    notes, index = dropNA(notes)
    for note in notes:
        note = re.sub("[^a-zA-Z0-9]", " ", note)

    lab_status = get_column_value(dataset, 'Lab Status')  # label
    status_num = copy.deepcopy(lab_status)
    for idx, item in enumerate(lab_status):
        if item == 'Negative ID':
            status_num[idx] = 0
        elif item == 'Positive ID':
            status_num[idx] = 1
        else:
            status_num[idx] = -1

    corpus = np.array(notes)
    label = status_num[index]
    unk = []
    for idx, lab in enumerate(label.tolist()):
        if lab == -1:
            unk.append(idx)

    vectorizer = TfidfVectorizer()
    result = vectorizer.fit_transform(corpus[unk]).toarray()
    # print(vectorizer.get_feature_names())
    print(result.shape)

    label = np.array(label)
    X_train, X_test, y_train, y_test = train_test_split(
        result, label[unk].astype('float64'), test_size=0.33, random_state=42)

    # Training the classifier & predicting on test data
    classifier = MultinomialNB()
    classifier.fit(X_train, y_train)

    y_pred = classifier.predict(X_test)
    # proba = classifier.predict_log_proba(X_test)
    # print(X_train)
    report = classification_report(y_test, y_pred)
    print('\n Accuracy: ', accuracy_score(y_test, y_pred))
    print('\nClassification Report')
    print('======================================================')
    print('\n', report)


if __name__ == "__main__":
    main()
