import sys
import pandas as pd
import sqlalchemy


def load_data(messages_filepath, categories_filepath):
    """
    Load data from the message and categories csv files.
    Input:
        messages_filepath: File path to messages csv
        categories_filepath: File path to categories csv
    Output:
        df: Merged dataframe using messages and categories data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = pd.merge(messages, categories, on="id")
    return df


def clean_data(df):
    """
    Function to clean data.
    Input:
        df: Dataframe to clean
    Output:
        df: Cleaned dataframe
    """
    categories = df['categories'].str.split(";", expand=True)
    row = categories.iloc[0]
    row = list(row)
    category_colnames = [row[i][:-2] for i in range(len(row))]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    df = df.drop(columns=['categories'])
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df


def save_data(df, database_filename):
    """
    Function to save the db file
    Input: 
        df: Dataframe to be saved
        database_filename: db file name
    """
    engine = sqlalchemy.create_engine('sqlite:///' + database_filename)
    df.to_sql('disaster_clean_data', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
