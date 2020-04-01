import sqlite3

import pandas as pd
def load_data(messages_filepath, categories_filepath):
    '''
    This function is to load input data from csv files
    :param messages_filepath:  mesasge.csv file path
    :param categories_filepath: category.csv file path
    :return: dataframe consist of merged data between two input files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = messages.merge(categories,on="id")
    
    return df
    
def clean_data(df):
    '''
    It take dataframe and clean it
    :param df: dataframe
    :return: cleaned dataframe for the modeling part
    '''
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(";",expand=True)   
    # select the first row of the categories dataframe
    row = categories.loc[:0,:]

    category_colnames = row.values.tolist()[0]
    # rename the columns of `categories`
    categories.columns = category_colnames
    
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]    

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # drop the original categories column from `df`
    df.drop(["categories"],axis=1,inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1)
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    return df


def save_data(df, database_filename):
    '''
    save dataframe to database
    :param df: dataframe
    :param database_filename: database file path
    :return: No return
    '''
    conn = sqlite3.connect(database_filename)
    df.to_sql('table1', conn, index=False)


def main():
        messages_filepath, categories_filepath, database_filepath = '../data/disaster_messages.csv', '../data/disaster_categories.csv', '../data/DisasterResponse1.db'
        

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print("DF")
        print(df.head())
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    



if __name__ == '__main__':
    main()