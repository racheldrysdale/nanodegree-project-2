import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    Takes the messages and categories datasets and returns a merged dataframe.
    
    INPUT:
    messages_filepath - (str) filepath of the messages CSV file
    messages_filepath - (str) filepath of the categories CSV file
    
    OUTPUT:
    df - (pandas dataframe) merged dataframe of the two datasets, joined on "id"
    '''

    # Load the datasets
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # Merge the datasets into a single dataframe
    df = messages.merge(categories, on='id')

    return df


def clean_data(df):
    '''
    Cleans and reformats the input dataframe including splitting out the message categories and dropping duplicates.
    
    INPUT:
    df - (pandas dataframe) input dataframe to be cleaned and reformatted
    
    OUTPUT:
    df - (pandas dataframe) cleaned dataframe with duplicates dropped, column added for each category, and unusual values dealt with
    '''

    # Create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)

    # Use the first row of the dataframe to extract a list of new column names for categories
    row = categories.head(1)
    category_colnames = []
    for i in range(0, 36):
        category_colnames.append(row[i][0][:-2])
    print(category_colnames)

    # rename the columns of `categories`
    categories.columns = category_colnames

    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # Change cases in the `related` column of 2s to 1s
    categories = categories.where(categories < 2, other=1)

    # Replace categories column in df with new category columns
    df = df.drop(columns='categories')
    df = pd.concat([df, categories], axis=1)

    # Drop any duplicates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    '''
    Saves clean dataframe to an SQLite database.
    
    INPUT:
    df - (pandas dataframe) dataframe to be saved
    database_filename - (str) name the database should be saved under
    '''

    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('disaster_response', engine, if_exists="replace", index=False)


def main():
    '''
    Script to execute importing of data, cleaning of dataframe, and saving of SQLite database.
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
