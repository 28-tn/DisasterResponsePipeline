import sys
import pandas as pd
import sqlite3
import numpy as np

def load_data(messages_filepath, categories_filepath):
    """
    Reads in messages and categorie data from corresponding csv files,
    merges the data to a single dataframe and returns the dataframe.

    Parameters
    ----------
    messages_filepath : STRING
        path to csv file with messages-data
    categories_filepath : STRING
        path to csv file with category-data

    Returns
    -------
    df: Pandas.DataFrame
        dataframe with merged data from messages and categories datasets
    """
    
    # Read in data
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge data
    df = messages.merge(categories, on='id', how='right')
    
    # Split categories into separate category columns
    category_columns = df['categories'].str.split(';', expand=True)
    category_colnames = category_columns.loc[0,:].str.split('-', expand=True).loc[:,0]
    category_columns.columns = category_colnames
    
    # Convert category values to numbers (0 or 1)
    for column in category_columns:
        # set each value to be the last character of the string
        category_columns[column] = category_columns[column].str[-1:]
        
        # convert column from string to numeric
        category_columns[column] = category_columns[column].apply(lambda x: int(x))
    
    # Concatenate the new columns to df and drop the old categories column
    df = df.drop(columns='categories')
    df = pd.concat([df, category_columns], axis=1)
    
    return df

def clean_data(df):
    """
    Cleans the data (removes duplicates and columns without variation)

    Parameters
    ----------
    df : Pandas.DataFrame
        Dataset to be cleaned

    Returns
    -------
    df : Pandas.DataFrame
        Cleaned dataset

    """
    
    # Drop duplicates
    old_lines = df.shape[0]
    df = df.drop_duplicates(subset='id')
    new_lines = df.shape[0]
    print('Removed {} duplicate rows'.format(old_lines-new_lines))
    
    # Drop columns without variation
    old_cols = df.shape[1]
    columns_without_variation = []
    for col in df.columns:
        if len(df[col].unique()) < 2:
            columns_without_variation.append(col)
    df = df.drop(columns=columns_without_variation)
    new_cols = df.shape[1]
    print('Removed {} columns without variation'.format(old_cols-new_cols))
    
    return df


def save_data(df, database_filename):
    """
    Save the dataframe to a sqlite3 database

    Parameters
    ----------
    df : Pandas.DataFrame
        Data to be saved in the database
    database_filename : String
        Path to the database

    Returns
    -------
    None.

    """
    
    # Connect to database
    con = sqlite3.connect(database_filename)
    cur = con.cursor()
    
    # Drop table if it exists
    cur.execute("DROP TABLE IF EXISTS response_data;")
    
    # Create new table
    sql_types = {np.dtype('object'): 'TEXT', np.dtype('int64'): 'INT'} # translates dtype to sql type
    create_cmd = 'CREATE TABLE response_data ('
    for col in df.columns:
        create_cmd += col+' '+sql_types[df.dtypes[col]]+', '
    create_cmd += 'PRIMARY KEY(id));'
    cur.execute(create_cmd)
    
    # Prepare  INSERT command
    insert_cmd = 'INSERT INTO response_data ('
    for col in df.columns:
        insert_cmd += col+', '
    insert_cmd = insert_cmd[:-2] # get rid of the last comma+space
    insert_cmd += ') VALUES ('
    for col in df.columns:
        insert_cmd += '?, '
    insert_cmd = insert_cmd[:-2] # get rid of the last comma+spac
    insert_cmd += ');'
    
    # Iter rows and insert data
    for index, data in df.iterrows():
        try:
            cur.execute(insert_cmd, data)
        except sqlite3.OperationalError:
            print('SQL error on:')
            print(insert_cmd)
    
    # Commit changes and close connection
    con.commit()
    con.close()


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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()