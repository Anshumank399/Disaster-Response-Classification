import sys
import pandas as pd
import re
import sqlalchemy
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import joblib


def load_data(database_filepath):
    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_clean_data", con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    text = re.sub('[^A-Za-z0-9]', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(RandomForestClassifier()))
            ])

    parameters = {
            'clf__estimator__n_estimators': [5, 40, 80, 100, 120, 160, 200],
            'clf__estimator__criterion': ["gini", "entropy"],
            'clf__estimator__max_depth': [5, 10, 15, 20, None],
            'clf__estimator__max_features': ["sqrt", "log2"],
            'clf__estimator__n_jobs': [-1]
            }

    clf_grid_model = RandomizedSearchCV(pipeline, parameters, cv=3)
    return clf_grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X,
                                                            Y,
                                                            test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
