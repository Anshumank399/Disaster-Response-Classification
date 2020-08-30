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
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from joblib import dump

def load_data(database_filepath):
    """
    Fuction to load data from the db file
    Input:
        database_filepath: db file name along with path
    Output:
        X: The feature attribute - message
        Y: To be predicted values.
        category_names: Name of columns
    """
    # load data from database
    engine = sqlalchemy.create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table("disaster_clean_data", con=engine)
    X = df['message']
    Y = df.drop(columns=['id', 'message', 'original', 'genre'])
    category_names = list(df.columns[4:])
    return X, Y, category_names


def tokenize(text):
    """
    Function to tokenize the text data.
    Input:
        text: Input text to be tokenized and cleaned.
    Output:
        clean_tokens: Clean text
    """
    text = re.sub('[^A-Za-z0-9]', ' ', text)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens


def build_model():
    """
    Fuction to build model using pipeline, parameter tuning and RandomizedSearchCV
    Output: 
        clf_grid_model: model
    """
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('clf', MultiOutputClassifier(AdaBoostClassifier()))
            ])

    parameters = {
            'vect__stop_words': ['english',None],
            'tfidf__smooth_idf': [True, False],
            'tfidf__norm': ['l2','l1'],
            'clf__estimator__learning_rate': [0.5, 1, 2],
            'clf__estimator__n_estimators': [20, 60, 100]
            }

    clf_grid_model = RandomizedSearchCV(pipeline,
                                        parameters,
                                        cv=3,
                                        refit=True,
                                        verbose=10,
                                        n_jobs=-1)
    return clf_grid_model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluate and print the model performance
    """
    y_pred = model.predict(X_test)
    # print the metrics
    for i, col in enumerate(category_names):
        print('{} category metrics: '.format(col))
        print(classification_report(Y_test.iloc[:,i], y_pred[:,i]))


def save_model(model, model_filepath):
    """
    Function to save the model
    """
    dump(model, model_filepath)


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
