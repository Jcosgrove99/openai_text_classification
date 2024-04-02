import pandas as pd
import numpy as np
from openai import OpenAI # openai version: 1.6.1
import tiktoken 
import re # package for removing special characters strings 
import json
import random
from openai import AsyncOpenAI 
import asyncio 
import sys 
import datetime as datetime
import fsspec
import s3fs
import time
import os

########################################### Pre API Call Processing Functions  ##############################################################################


# helper functions to count number of tokens in each text value 
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def remove_special_characters(text):
    """Returns only the non special characters within a string"""
    text = str(text)
    pattern = r'[^a-zA-Z0-9\s]' # This pattern will remove anything that is not alphanumeric or whitespace
    clean_text = re.sub(pattern, '', text)
    return clean_text


def collapse_text_values_to_csv_groups(data, text_variable, tokens_per_group): 
    """Groups together the text values into groups no bigger then the tokens per groups
        Must already have `text_value_id` column"""
    
    print("collapsing raw text values into .csv groups for api call")

    # remove special characters from the text data 
    data['text_values_clean'] = data[text_variable].apply(lambda x: remove_special_characters(x))

    # create a column with the count of tokens in each text value 
    data['token_count'] = data["text_values_clean"].apply(lambda x: num_tokens_from_string(x, "cl100k_base")) 

    # create a column that is the cumulative sum of the token column 
    data['token_count_cumsum'] = data['token_count'].cumsum()

    # use the interger division operator to create api call groups 
    data['api_call_group'] = data['token_count_cumsum'] // tokens_per_group

    # add id to text value 
    data['text_values_clean'] = data['text_value_id'].astype(str) + " " + data['text_values_clean'] 

    # keep variable(s) to be used to be feed into chatgpt
    prep_data = data[["text_values_clean", "api_call_group"]]

    # group by text values 
    prep_data = prep_data.groupby('api_call_group').agg({'text_values_clean': lambda x: ', '.join(map(str, x))})

    return prep_data 

def create_category_id_key(categories_df): 
    
    print('creating category id key to keep track of categories gpt chooses')
    categories_df = categories_df.reset_index()

    # create a unique id for each category (always 2 digit number)
    categories_df['category_id'] = categories_df.index + 10

    # TODO: add a warning if the number of categories exceeds 90
    return categories_df

def create_system_instructions(gpt_directions, category_id_key, other_category): 
    """takes in the gpt_directions defined by the user and the category_id_key defined earlier"""
    print('bind together system instructions based on user inputs')
    
    if other_category == True: 
        other = "If you are uncertain, then choose 99 for the 'Other' category.\n"
        other_category = "\n - Other (99)\n"
    else: 
        other = "The 'Other' category is not applicable for this prompt. Please select the most appropriate category from the provided options.\n"
        other_category = ""
    # turn categories list into one string with the category id attatched 
    # attatched the unique id for each category to the end of the category string 
    category_id_key['category_clean'] = "- " + category_id_key['category'] + " (" + category_id_key['category_id'].astype(str) + ")"

    # select only category clean 
    category_id_key = category_id_key[['category_clean']]

    # collapse categories down into one string value 
    category_id_key['group'] = 1 
    categories_string = category_id_key.groupby('group').agg({'category_clean': lambda x: '\n '.join(map(str, x))})
    categories_string = categories_string['category_clean'][1]

    json_request = '''You will output this as a JSON document {expense id:category id number}\n'''
    
    # Join together entire system instructions
    system_instructions = gpt_directions +  other + json_request + categories_string + other_category

    return system_instructions

########################################### Post API Call Processing Functions  ##############################################################################

def extract_raw_logprob_data(logprobs_content, output):

     # Create empty lists to be filled with the logprob content 
    token_list = []
    logprob_list = []  

    # loop through logprobs content, each item is a list for each token
    for item in logprobs_content: 
        # select the top log probs section which contains the data we want for each token 
        topLogprobs = item.top_logprobs[0]
        # loop through each top log probs list for each token
        for topLogporob in item.top_logprobs: 
            # extract the token and log prob associate with it
            TOKEN = topLogporob.token
            LOGPROB = topLogporob.logprob
            # append these values to a list 
            token_list.append(TOKEN)
            logprob_list.append(LOGPROB)
    
    # after creating token and log prob lists, turn it into a dataframe  
    logprobs_df = pd.DataFrame({
        "token":token_list,
        "logprob":logprob_list, 
        # saving model used and system fingerprint 
        "model":output.model, 
        "system_fingerprint":output.system_fingerprint, 
        "completion_tokens":output.usage.completion_tokens,
        "prompt_token":output.usage.prompt_tokens,
        "total_tokens":output.usage.total_tokens,
        })
    
    return logprobs_df


def remove_excess_logprob_text(text):
    """Returns only the non special characters within a string"""
    text = str(text)
    pattern = r'[^0-9:]' # This pattern will remove anything that is not alphanumeric or whitespace
    clean_text = re.sub(pattern, '', text)
    return clean_text


# clean logprob data 
def clean_raw_logprob_data(logprobs_df): 

    # remove special characters from the text data 
    logprobs_df['token_clean'] = logprobs_df['token'].apply(lambda x: remove_excess_logprob_text(x))

    # assign NA to empty tokens 
    logprobs_df['token_clean'] = np.where(logprobs_df['token_clean']=='', np.nan, logprobs_df['token_clean'])

    # drop empty tokens returned 
    logprobs_df = logprobs_df.dropna(subset=['token_clean'])

    # turn clean tokens into a string column 
    logprobs_df['token_clean'] = logprobs_df['token_clean'].astype(str)

    # TODO: important assumptions here. Double check that this is the best way to do things. 
    ### Identify which values returned in logprobs are the text_value_ids vs. category_ids from gpt ###########

    # text_value_ids are 6 digit numbers, and gpt returns them as 2 tokens of 3 numbers. I identify them by finding where character value is 3. 
    logprobs_df["id_type"] = np.where(logprobs_df["token_clean"].str.len() == 3, "text_value_id", np.nan)

    # category_ids are 2 digit numbers, and gpt returns them as 1 token of 2 numbers. I identify them by finding where character value is 2. 
    logprobs_df["id_type"] = np.where(logprobs_df["token_clean"].str.len() == 2, "category_id_gpt",logprobs_df["id_type"])

    # because its a json file, the text_value_ids assiocitate with the category_id are seperated by a ":"
    # I identify them by finding where character value is 1. 
    logprobs_df["id_type"] = np.where(logprobs_df["token_clean"].str.len() == 1, "divider",logprobs_df["id_type"])

    # reset index
    logprobs_df = logprobs_df.reset_index()

    # TODO: another important assumption to double check that this holds true
    # after cleaning the logprobs are in descending rows where each four rows are associated together. 
    # for example (row 1 = text_value_id first 3 digits, row2 = text_value_id last 3 digits, row3 = divider, 
    # row4 = category_id that gpt assigned the text_value_id)
    logprobs_df['group'] = logprobs_df.index // 4   

    return logprobs_df


def create_final_logprob_data(clean_logprob_data): 

    # TODO: for developing, only selecting necesary columns 
    clean_logprob_data = clean_logprob_data[['token_clean', 'id_type', 'group', 'logprob']]
    
    # TODO: groupby id_type and group. concate the token_clean column, select the first value of all other columns 

    # Making sure i am grouping by two columns correctly 
    clean_logprob_data = clean_logprob_data.groupby(['id_type', 'group']).agg({

            'token_clean': lambda x: ''.join(map(str,x)), 

            # selecting first logprob, we only care about the category_id logprob which will only have 1 value. 
            # the text_value_ids with 2 logprobs we do not care about. 
            'logprob': "first"
    }
    )
    
    # reset index
    clean_logprob_data = clean_logprob_data.reset_index()

    # pivot wider according to group number
    logprob_data_wide = clean_logprob_data.pivot_table(index = 'group', columns = 'id_type', values = ['logprob' ,'token_clean'], aggfunc = "first")

    # flattened the grouped dataframe
    logprob_data_wide.columns = logprob_data_wide.columns.to_series().str.join('_')

    logprob_data_wide = logprob_data_wide.reset_index()

    # select only desired columns 
    logprob_data_wide = logprob_data_wide[['logprob_category_id_gpt', 'token_clean_category_id_gpt', 'token_clean_text_value_id']]

    # rename columns 
    logprob_data_wide.rename(columns={'token_clean_category_id_gpt': 'category_id_gpt', 'token_clean_text_value_id':'text_value_id'}, inplace=True)

    return logprob_data_wide

 ########################################### A-synchronous API call functions ##############################################################################


# I think to implement this asynchronous work, we would have to save the output of the asynch function. 
# and then once its complete, download those files back in and continue with the program. 

def bind_csv_files(folder_path):

    print(f'Binding .csv files from {folder_path}')

    # Get list of all files in the folder
    files = os.listdir(folder_path)
    
    # Filter out only CSV files
    csv_files = [file for file in files if file.endswith('.csv')]
    
    # Initialize an empty list to store dataframes
    dfs = []
    
    # Read each CSV file and append its dataframe to the list
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        dfs.append(df)
    
    # Concatenate all dataframes in the list along rows
    concatenated_df = pd.concat(dfs, ignore_index=True)
    
    print(f'Finished binding .csv files from {folder_path}')

    return concatenated_df


