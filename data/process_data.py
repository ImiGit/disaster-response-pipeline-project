import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    This function loads the data from two csv filepaths and merges them into
    one pandas dataframe.

    Parameters:
        messages_filepath: This is the filepath to the messages csv file which contains the messages
        categories_filepath: filepath to categories csv file which holds category of each message

    Returns:
        panda's dataframe: The merged dataframe of the loaded data from two file pathes is returned.
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages.merge(categories, how="outer", on="id")


def clean_data(df):
    '''
    This function cleans the input dataframe and returns it.

    Categories are all saved in one column. So, first these categories are split and then
    only the their names are used as columns names and their last digit values are used
    to make a binary table.
    Duplicate values are droped.
    One column, "child_alone", is all zeros and will be dropped.
    There are 188 2s in "related" column. They will be replaced with 1.

    Parameters:
        df: A panda's dataframe object which needs to be cleaned.

    Returns:
        df: The cleaned dataframe.
    '''
    categories = df["categories"].str.split(pat=";", expand=True)
    row = categories.iloc[1,:]
    category_colnames = [(lambda x: x[:-2])(x) for x in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df.drop("categories", axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)

    df.drop_duplicates(keep="first", inplace=True)
    df.drop("child_alone", axis=1, inplace=True)
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)

    return df


def save_data(df, database_filename):
    '''
    This function saves the input panda's dataframe object into a sqlite database file.

    Parameters:
        df: The panda's dataframe object that we like to save.
        database_filename: The desired file name of the sqlite database file.

    Returns:
        panda's dataframe: The merged dataframe of the loaded data from two file pathes is returned.
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('messages', engine, index=False, if_exists='replace')


def main():
    '''
    This is the main function.

    First it checks if the input argument on the system is correct and messages and categories
    filepaths are given to load the data and database filepath is given to save the data later
    into a sqlite database.

    If they are all given, it first loads the data, then cleans it, and finally saves it.
    Otherwise, it prints a message informing the user that not all the needed arguments are
    provided correctly.
    '''
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
