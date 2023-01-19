# import libraries
import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """Load and merge the messages and categories datasets.

    Args:
    messages_filepath: str. the filepath for the messages dataset, ex:'disaster_messages.csv'
    categories_filepath: str. the filepath for the categories dataset, ex:'disaster_categories.csv'

    Returns:
    merged_df: dataframe. the loaded messages with its corresponding categories 
    """
    messages = pd.read_csv(messages_filepath) # load messages dataset
     
    categories = pd.read_csv(categories_filepath) # load categories dataset
      
    merged_df = messages.merge(categories) # merge both datasets
    
    return merged_df

def clean_data(df):
    """Preprocess and clean the dataset.

    Args:
    df: dataframe. the loaded dataset which conatins messages with its corresponding categories 

    Returns:
    cleaned_df: dataframe. the cleaned dataset
    """
    
    # Split categories into separate category columns
    categories = df['categories'].str.split(";", expand=True)   # create a dataframe of the 36 individual category columns
    row = categories.iloc[0]
    category_colnames = [cat[:-2] for cat in row.values]   # Extract the names of the new category columns
    categories.columns = category_colnames
    
    # Convert the category values to numbers (0 or 1)
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)
    
    # Replace any 2 to 1 in the 'related' category
    categories['related']=categories['related'].replace(2, 1)
    
    # Replace categories column in df with the new category columns
    df.drop('categories', axis=1, inplace=True)         
    cleaned_df = pd.concat([df, categories], axis=1)

    cleaned_df.drop_duplicates(inplace=True)   # Remove duplicate rows from the cleaned dataset
    
    return cleaned_df

def save_data(df, database_filename):
    """Save the clean dataset into a sqlite database.
    
    Args:
    df: dataFrame. the cleaned data
    database_filename: str. the filepath of the database to save the cleaned data, ex:'DisasterResponse.db'
    
    Returns:
    Saves the clean dataset into a sqlite database
    """
    
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('DisasterResponse', engine, index=False)

def main():
    """The main function to implement the ETL pipeline    
    This function will load the dataset from the input files, Preprocess and clean it, then save the cleaned data in a sqlite database
    """

    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        original_df = load_data(messages_filepath, categories_filepath)  # Load the dataset from the provided filepaths

        print('Cleaning data...')
        clean_df = clean_data(original_df)   # Preprocess and clean the dataset
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(clean_df, database_filepath)   # Save the clean dataset into a sqlite database
        
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