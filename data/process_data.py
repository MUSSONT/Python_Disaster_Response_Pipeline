import sys
import pandas as pd
import sqlalchemy
from sqlalchemy import engine, create_engine

def load_data(messages_filepath, categories_filepath):
    """generates dataframe from the 2 data sources, messages and catgories
    
    Args -
    messages_filepath 
    categories_fielpath
    
    returns
    Merges above sources into one dataframe"""
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories, on='id', how='outer')
    return df

def clean_data(df):
    """function to clean data,drops dupes, prepares needed columns
    
    Args:
    df - dirty data frame parsed in
    Returns :
    cleaned df"""
    categories = df['categories'].str.split(';', expand=True)
    category_colnames  =[ x[:-2] for x in categories.iloc[0]]
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
        categories[column] = categories[column].str[-1:]
    
    # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
        
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df,categories], axis=1)
    df.drop_duplicates(inplace=True)
    return df

def save_data(df, database_filename):
    """saves df to a sql table 
    
    Args:
    Df - the dataframe for saving
    database_filename - path for db to save to
    
    Returns: nothing """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('CleanMessages2', engine, index=False)
    pass  


def main():
    """main function to go through steps of parsing through each func"""
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