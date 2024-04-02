#
# Purpose of Script: Create code to be reused for various text classifcation needs using openai 
#                     api model gpt4 
# Author: Jake Cosgrove 
# Date: 03-29-2
# 
# Order of Operations: 
# 1. User defines constants 
# 2. Perp data and prompt for api call 
# 3. Call openai api chat completion endpoint 
# 4. process output from api call 
# Resources: https://huggingface.co/blog/Andyrasika/logprobs-transformers
#. The best strategy is narrowing functions to just use their parameters, and this also makes the functions relatively easy to test, as we have seen with Doctests

# Outstanding TODO 04-01-2024 ----------------------------------------------------------------------------------------------------
# - double check log prob processing [checked]
# - add in print statements to help with debugging 
# - add in assert statements to prevent mistakes
# - ask openai engineer if I am handling the log probabilities properly
# - why did some logprobs start coming as NaN? 
# - I think I should just save the entire json file that is returned and then process it outstide of the async. 

# Load in necessary libraries
import pandas as pd
import numpy as np
from openai import AsyncOpenAI # specific function for synch api calls
import tiktoken 
import re # package for removing special characters strings 
import json
import random
from text_classification_helper_functions import * # make sure this file is in the working directory. 
from datetime import datetime
import pprint
from IPython.display import display, HTML
import os
pd.options.mode.chained_assignment = None # silance pandas warnings 

##################################### User Defined Constants #########################################################################

# RAW_DATA_PATH = path to data set the requires text classification. Must have at least 1 descriptive column. (Ex. Expense_Description) 
# TEXT_VARIABLE = variable with information that gpt will use to choose the category it belongs to (Often a text value describing the event)
# TOKENS_PER_GROUP = Defines the number of tokens in each prompt that will be feed into gpt4. Higher tokens means less api calls, but each api 
#                    call will take longer. 
# API_KEY = your openai api key 
# GPT_DIRECTIONS = This is instructions to the model on how you want it to behave. 
# OTHER_CATEGORY = Boolen value. If True then the model will assign "other" to any values it's uncertain about. 
# TODO: for the categories list, we should include a column of the variable name so that we can join it together to it at the end. 
# CATEGORIES = A list of the categories that you would like the model to use to categorize your data (Do not include an "Other" Category)

TEXT_VARIABLE = "desc"
TOKENS_PER_GROUP = 250
RAW_DATA_PATH = "/Users/jakecosgrove/Documents/open_ai_api/text_classification/code_template/multiple_api_call_template/data/Other_Expenses_GPT.csv"
API_KEY = ""

GPT_DIRECTIONS = f'''
You will receive a comma-separated list of expenses, each accompanied by a seven-digit description ID. 
Your task is to categorize each expense description into one of the provided expense categories based on their category ID.
Choose ONLY from the list of categories provided here. Choose ONLY one expense category per expense presented.
'''

OTHER_CATEGORY = True

# Load in a dataframe with two columns: Category Descriptions and variable names (Do Not Include an Other Category)
categories_data_path = "/Users/jakecosgrove/Documents/open_ai_api/text_classification/code_template/multiple_api_call_template/data/Expenditures_mapping.csv"
expense_categories = pd.read_csv(categories_data_path)
expense_categories = expense_categories.loc[expense_categories['Do Not Include'] != 'DO NOT INCLUDE']

# Make sure column names are 'category' and 'varname' 
expense_categories.rename(columns = {'Prompt Category':'category', 'src_varname':'varname'},inplace=True)
CATEGORIES = expense_categories[['category', 'varname']]

######################################################################################################################################################

##################################### Combine together system instructions for api call ##############################################################

# Accessing OpenAI with Open Research API key
openai_client = AsyncOpenAI(api_key = API_KEY)

# Add id numbers to the list of categories you want to use (will use this key at the end to match categories back) 
category_id_key = create_category_id_key(CATEGORIES)

# Create system instructions (aka prompt) 
system_instructions = create_system_instructions(GPT_DIRECTIONS, category_id_key, OTHER_CATEGORY)

# Look at what system instructions look like
pprint.pprint(system_instructions)

##################################### Load and prep raw data frame for api call #####################################################################

# Load in data
raw_data = pd.read_csv(RAW_DATA_PATH)

# Assign a id number to keep track of each row for when the rows get collapsed 
random.seed(42)
raw_data['text_value_id'] = np.random.randint(100000, 999999, size=len(raw_data))

# take raw data with text_value_id and collapse it into groups no bigger then the token per group
input_data = collapse_text_values_to_csv_groups(raw_data, TEXT_VARIABLE, TOKENS_PER_GROUP)
input_data = input_data.reset_index()

# Testing section: subset input data frame as needed
# input_data = input_data.iloc[:3]
# input_data = input_data.reset_index()

########################### Creating Async Functions to Call OpenAI ChatCompletion ############################################################

async def call_api(prompt, group, system_instructions, output_folder, client): 
  
  # tracking the time the call starts 
  now = datetime.now().strftime("%H:%M:%S")
  print("Called api for group " + str(group) + " at " + str(now))

  # api call used within an await (await is a asynch function that allows for other code to be ran while waiting for this response)
  output = await client.chat.completions.create(
      seed = 42, # set random seed for "attempted" consistent responsesd 
      temperature = 0, # Controls randomness (0 - 2) 0 is more deterministic, 2 is more random
      model = "gpt-4-1106-preview",
      response_format={ "type": "json_object" }, # max tokens returned in json mode == 4096
      logprobs = True, # returns the probability that the models has for each tokens
      top_logprobs = 1, # tells it how many of the top responses to send back 
      messages = [
          {"role":"system",
           "content":system_instructions},
  
          {"role":"user",
           "content": prompt}]
      ) 
  
  ### Processing Model Output #######

  # Processing the chat completion output 
  finish_reason = output.choices[0].finish_reason

  # parse out response from output
  content = output.choices[0].message.content
  # turn response into a json    
  content_json = json.loads(content)
  # turn response into a dataframe 
  categories_wide = pd.json_normalize(content_json)
  categories_final = categories_wide.T
  categories_final = categories_final.reset_index()
  categories_final.rename(columns={0: 'category_id_gpt', 'index':'text_value_id'}, inplace=True)

  # processing the chat completion log probabilities 
  logprobs_content = output.choices[0].logprobs.content

  # extract raw logprob data into dataframe 
  logprob_raw = extract_raw_logprob_data(logprobs_content, output)

  # clean the tokens returned and create columns identifying groups 
  logprob_clean = clean_raw_logprob_data(logprob_raw)
  # TODO: save meta data will have to take place during the asnch functions somehow. 

  # pivot wider the raw data and keep the logprobs for only the category id that gpt choose
  logprob_final = create_final_logprob_data(logprob_clean) 

  #### Create & Save API Output Dataframe by Groups ####

  # prep columns to be joined 
  categories_final['text_value_id'] = categories_final['text_value_id'].astype(int)
  logprob_final['text_value_id'] = logprob_final['text_value_id'].astype(int)

  # drop category_id_gpt because it's already in categories_final
  logprob_final = logprob_final[['logprob_category_id_gpt', 'text_value_id']]

  # join the logprobs of the categories to the output data 
  api_output_data = pd.merge(categories_final, logprob_final, how='left', on='text_value_id')

  # add in api_call_group column 
  api_output_data['api_call_group'] = group

  # create file name 
  file_name = "api_output_data_group" + str(group) + "_.csv"

  # define folder name created earlier
  api_output_folder = output_folder + "/api_output_raw_data/"

  # save api output to newly create api output folder 
  api_output_data.to_csv(api_output_folder + file_name)

  print("Saved response for group " + str(group) + ". (Finish reason:" + finish_reason + ")")

  return 

#########################Defining Main Async Function to Allow for Asynchornous API Calls ##########################################################################################################

# loops through rows of the grouped data and calls the api for each row 
async def main(data, system_instructions, output_folder, client):

  print("creating api output raw data folder") 
  # create folder for api output data to live temporarily before binding together to output_data 
  api_output_folder = output_folder + "/api_output_raw_data"
  os.makedirs(api_output_folder)
  
  tasks = [call_api(prompt = row['text_values_clean'], group = row['api_call_group'], 
                    system_instructions = system_instructions, 
                    output_folder = output_folder, 
                    client = client,
                    ) for index, row in data.iterrows()]
  
  # wait for all api calls to complete before continuing 
  response = await asyncio.gather(*tasks)

  print("COMPLETED ALL API CALLS")

  # bing together all .csv files in api_output_raw_data folder 
  output_data = bind_csv_files(folder_path = api_output_folder)

  # return binded together output data frame
  return output_data

# create output folder 
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_folder_name = f"output_{timestamp}"
os.makedirs(output_folder_name) 

# execute api calls all at once 
output_data = asyncio.run(main(data = input_data, 
                 system_instructions = system_instructions, 
                 client = openai_client, 
                 output_folder = output_folder_name))

# check that output is the same length as the input 
if len(output_data) != len(raw_data): 
   message = "WARNING! output_data is different length then raw_data"
   print(message)

########################################### Create Final Results Data Frame #########################################################################

# Join output data to category id key first 
output_data['category_id_gpt'] = output_data['category_id_gpt'].astype(int)
category_id_key['category_id'] = category_id_key['category_id'].astype(int)

# Add Other row to category key for left join 
other_row = {'category': 'other', 'varname': 'Other', 'category_id': 99}
category_id_key = category_id_key._append(other_row, ignore_index=True)

# join ouput data with category key 
ouput_data_with_category = pd.merge(output_data, category_id_key, how='left', left_on='category_id_gpt', right_on='category_id')

if len(ouput_data_with_category) != len(raw_data): 
   message = "WARNING! output_data_with_category is different length then raw_data"
   print(message)

# Before saving check their are no NA values for category output 
if ouput_data_with_category['category'].isna().any():
    print("Warning: The column '{}' contains NA values. Script halted.".format('category_gpt'))
    exit() 

# select only final columns for output_data 
ouput_data_with_category = ouput_data_with_category[['text_value_id', 'varname', 'category_id_gpt', 'category', 'logprob_category_id_gpt']]

# rename with gpt ending 
ouput_data_with_category.rename(columns={'varname':'varname_gpt', 'category':'category_gpt'}, inplace=True)

# join new final columns to original raw data 
raw_data['text_value_id'] = raw_data['text_value_id'] .astype(int)
ouput_data_with_category['text_value_id'] = ouput_data_with_category['text_value_id'].astype(int)

# join the new gpt categories back to the raw data 
output_data_final = pd.merge(raw_data, ouput_data_with_category, how='left', on='text_value_id')

if len(output_data_final) != len(raw_data): 
   message = "WARNING! output_data_final is different length then raw_data"
   print(message)

# TODO: join in meta data from api call 

# Save final output data with three new columns, `category_logprob_gpt`, `category_gpt`, `varname_gpt`
output_data_final.to_csv(output_folder_name + "/output_data_final.csv")

# Check for any NA's in 

# save the prompt used 
text_path = output_folder_name + "/system_instructions.txt"
with open(text_path, "w") as file: 
    file.write(system_instructions)




  





