"""
Sample Execution:
> python process_data.py disaster_messages.csv disaster_categories.csv DisasterResponse.db

Args:
    1. Path to the messages CSV file
    2. Path to categories CSV file
    3. Path to SQLite output database
"""

# Import all libraries
import sys
import pandas as pd
from sqlalchemy import create_engine
 
def load_data(messages_filepath, categories_filepath):
    """
    INPUT:
        messages_filepath: Path to the messages CSV file
        categories_filepath: Path to the categories CSV file
    OUTPUT:
        df: combined dataframe of messages & categories
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    """
    INPUT:
        df: combined dataframe of messages & categories
    OUTPUT:
        df: cleaned dataframe of messages & categories
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(pat=';',expand=True)
    
    #Fix the categories columns name
    row = categories.iloc[[1]]
    category_colnames = [category_name.split('-')[0] for category_name in row.values[0]]
    categories.columns = category_colnames
    
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x.split('-')[-1])
        categories[column] = categories[column].astype(int)
        
    # drop the original categories column from `df`
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    df = df.drop_duplicates()
    
    return df

def save_data_to_db(df, database_filename):
    """
    INPUT:
        df -> cleaned dataframe of messages & categories
        database_filename -> Path to SQLite output database
    """
    
    engine = create_engine('sqlite:///'+ database_filename)
    table_name = database_filename.replace(".db","") + "_table"
    df.to_sql(table_name, engine, index=False, if_exists='replace')

def main():
    """
    Main function which will kick off the data processing functions. There are three primary actions taken by this function:
        1) Load Messages Data with Categories
        2) Clean Categories Data
        3) Save Data to SQLite Database
    """
    
    # Execute the ETL pipeline if the count of arguments is matching to 4
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:] # Extract the parameters in relevant variable

        print('Loading messages data from {} ...\nLoading categories data from {} ...'
              .format(messages_filepath, categories_filepath))
        
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...')
        save_data_to_db(df, database_filepath)
        
        print('Cleaned data has been saved to SQLite db: {}'.format(database_filepath))
    
    else: 
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()