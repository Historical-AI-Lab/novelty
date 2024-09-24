#!/usr/bin/env python
# coding: utf-8

# In[549]:


#imports
import seaborn as sns
import requests
import time
import string
import unicodedata
import pprint
import numpy as np
import spacy

import pandas as pd
df_2 = pd.read_csv('new_labels_aug22.csv')


# In[550]:


#read in our data
df = pd.read_csv('result_df_july18_1136am.csv')


# For this specific subsample example, we need to add back S2 metadata like S2 titles and pubdates (there were left out/removed somehow when this VIAF subsample was created

# In[551]:


#for this specific subsample example, we need to add back S2 info like S2 titles and pub dates

#add in the S2 titles

#get publication date to do avergae pub date work
# Load the metadata TSV file into a DataFrame
file_path = 'LitMetadataWithS2 (3).tsv'



import pandas as pd
meta = pd.read_csv(file_path, sep='\t')

import re
def normalize_text(text):
    """
    Normalize the text by converting to lowercase, preserving commas and spaces,
    and removing unnecessary special characters.
    """
    if pd.isna(text):
        return ''
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^\w\s,]', '', text)  # Remove all characters except alphanumeric, spaces, and commas
    return text.strip()  # Trim leading and trailing spaces

# In[552]:


# Ensure the authors column is of type list
meta['author'] = meta['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)

# Use the explode method to expand the authors column
meta_exploded = meta.explode('authors')

meta = meta_exploded

# Display the exploded DataFrame
# print(meta_exploded)


# In[553]:


for idx, row in meta.iterrows():
    author = str(row['author'])
    author_clean = normalize_text(author)
    meta.at[idx, 'author'] = author_clean


# Add back the S2 titles

# In[554]:


# Initialize an empty dictionary to store the author-title mapping
author_titles = {}

# Iterate over the DataFrame rows
for index, row in meta.iterrows():
    author = str(row['author'])
    title = row['S2titles']
    
    # Use setdefault to initialize the list if the author is not already in the dictionary
    author_titles.setdefault(author, []).append(title)

# # Display the dictionary to verify
# print(author_titles)


# In[555]:


import re


title_author_dict = author_titles
# Normalize the dictionary
#normalized_dict = {normalize_text(title): normalize_text(author) for title, author in title_author_dict.items()}
#normalized_dict = {normalize_text(title): normalize_text(author) for title, author in title_author_dict.items()}

# Normalize the DataFrame
# df['normalized_S2_titlelist'] = df['S2_Titlelist'].apply(normalize_text)
df['normalized_author'] = df['author'].apply(normalize_text)


# In[556]:


normalized_dict = {normalize_text(str(author)): normalize_text(str(title)) for title, author in author_titles.items()}


# In[557]:


swapped_dict = {v: k for k, v in normalized_dict.items()}


# In[558]:


# swapped_dict


# In[559]:


df['S2_Titlelist'] = df['normalized_author'].map(swapped_dict)


# In[560]:


# df


# Now add back the S2 pubyears

# In[561]:


# Initialize an empty dictionary to store the author-title mapping
author_years = {}

# Iterate over the DataFrame rows
for index, row in meta.iterrows():
    author = str(row['author'])
    year = row['year']
    
    # Use setdefault to initialize the list if the author is not already in the dictionary
    author_years.setdefault(author, []).append(year)

# Display the dictionary to verify
# print(author_titles)


# In[562]:


# normalized_dict = {normalize_text(str(author)): normalize_text(str(title)) for title, author in title_author_dict.items()}
normalized_dict = {normalize_text(str(author)): title for title, author in author_years.items()}


# In[563]:


#normalize then apply dictionary
# cleaned_dict = {key.strip("[]'"): value for key, value in normalized_dict.items()}
swapped_dict = {v: k for k, v in normalized_dict.items()}


def update_pubdates(name):
    if name in swapped_dict:
        years = swapped_dict[name]
        return years
    
df['S2_pubdates'] = df['normalized_author'].apply(update_pubdates)


# In[564]:


# df['S2_pubdates'] = df['author'].map(author_years)


# In[565]:


df['S2_pubdates']


# In[612]:





# In[566]:


# df


# Step 1: Create the Features We Need

# Get all of the necessary information/entries from VIAF
# (For the subsample example, this has already been done and can be skipped)

# First, set up the functions to search VIAF and intialize the API

# In[567]:


def get_name(name):
    params = {
        'query' : f'local.personalNames = "{name}"',
        'maximumRecords': 10,
        'startRecord' : 1,
        'sortKeys': 'holdingscount',
        'httpAccept': 'application/json'
    }

    headers={'User-Agent': user_agent}
    url = "https://viaf.org/viaf/search"

    r = requests.get(url,params=params,headers=headers, timeout=10)
    data = r.json()

    return data


# In[568]:


user_agent = 'YOUR PROJECT NAME HERE'
id_column_name = "Name"

pause_between_req = 1

use_title_reconcilation = True

cache = {}


# Then create a function to extract the information that we want out of the VIAF JSON returned

# In[569]:


# import pandas as pd
# from fuzzywuzzy import fuzz

# birthdates_df = df

# # Function to perform search and return DataFrame with search results
# def search_author(author_name):
#     # Initialize list to store search results
#     search_results = []

#     try:
#         data = get_name(author_name)
#         records = data['searchRetrieveResponse']['records']
#         total_records = len(records)

#         for idx, record in enumerate(records):
#             record_data = record['record']['recordData']
#             birthdate = record['record']['recordData']['birthDate']
#             viaf_title_list = []
#             # Extract titles if available
#             if 'titles' in record_data:
#                 titles_data = record_data['titles']
#                 if titles_data is not None:
#                     if isinstance(titles_data['work'], list):
#                         # Extract titles from the list of works
#                         viaf_title_list.extend([work['title'] for work in titles_data['work'] if 'title' in work])
#                     else:
#                         # Extract title from a single work
#                         title = titles_data['work'].get('title')
#                         if title:
#                             viaf_title_list.append(title)

# #             # Get S2 titles for the author
# #             s2_titles = birthdates_df.loc[birthdates_df['author'] == author_name, 'S2 Titles'].iloc[0]
# #             if isinstance(s2_titles, float):
# #                 s2_titles = []

# #             # Perform fuzzy title matching
# #             title_matched = False
# #             matched_title = None
# #             for title_from_viaf in title_list:
# #                 for s2_title in s2_titles:
# #                     if fuzz.partial_ratio(s2_title, title_from_viaf) >= 70:
# #                         title_matched = True
# #                         matched_title = s2_title
# #                         break
# #                 if title_matched:
# #                     break

# #             # Append author name, title list, S2 titles, title matched, matched title, and birthdate to search results
# #             # birthdate = birthdates_df.loc[birthdates_df['author'] == author_name, 'birthdate'].iloc[0]
# #             search_results.append({'author': author_name, 'record_count': len(records),
# #                                    'record_enumerated': idx, 'title_list': title_list,
# #                                    'S2 Titles': s2_titles, 'title_matched': title_matched,
# #                                    'matched_title': matched_title, 'birthdate': birthdate})
            
#             # Append author name, title list, S2 titles, title matched, matched title, and birthdate to search results
#             # birthdate = birthdates_df.loc[birthdates_df['author'] == author_name, 'birthdate'].iloc[0]
#             title_list = df.loc[df['author'] == author_name, 'title_list'].iloc[0]

#             search_results.append({'author': author_name, 'record_count': len(records),
#                                    'record_enumerated': idx, 'viaf_title_list': viaf_title_list,
#                                    'birthdate': birthdate,'title_list':title_list})
#     except Exception as e:
#         print(f'Processing Error for {author_name}: {e}')

#     return search_results

# # Iterate through each author name in the original DataFrame and perform the search
# all_search_results = []
# row_counter = 0  # Counter to limit the number of rows processed

# for author_name in birthdates_df['author']:
#     search_results = search_author(author_name)
#     all_search_results.extend(search_results)
#     # row_counter += 1
#     # if row_counter >= 5:  # Adjust the number as needed
#     #     break

# # Convert list of dictionaries to DataFrame
# all_search_results_df = pd.DataFrame(all_search_results)
# print(all_search_results_df)


# Rename some columns so they match later usage

# In[570]:


result_df = df
result_df['VIAF_titlelist'] = result_df['record_enumerated_titles'] 
result_df['S2_titlelist'] = result_df['S2titles']
result_df['average_S2_pubdate'] = result_df['avg_pubdates']
result_df['VIAF_birthdate'] = result_df['standard_birthdate']
result_df['S2_Author'] = result_df['author']


df = result_df


# In[571]:


df


# create our pubage variable

# In[628]:


#create our defined function variables
import pandas as pd
from ast import literal_eval

def find_max_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return max(pubdates)
    return None

def find_min_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return min(pubdates)
    return None
    
def find_avg_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return (sum(pubdates)/len(pubdates))
    return None
import ast
def birth2maxdate(birth, pubdates):
        # Convert the string representation to an actual tuple
        if isinstance(pubdates, str):
            pubdates_tuple = ast.literal_eval(pubdates)
        else:
            pubdates_tuple = pubdates
        # Check if it's a tuple of integers
        if isinstance(pubdates_tuple, tuple) and all(isinstance(date, int) for date in pubdates_tuple):
            max_pubdate = find_max_pubdate(pubdates)
            if max_pubdate is not None:
                birth2maxdate = max_pubdate - birth
                abs_birth2maxdate = abs(birth2maxdate)
                return birth2maxdate, abs_birth2maxdate
        return None, None

def birth2mindate(birth, pubdates):
    min_pubdate = find_min_pubdate(pubdates)
    if min_pubdate is not None:
        birth2mindate = min_pubdate - birth
        abs_birth2mindate = abs(birth2mindate)
        return birth2mindate, abs_birth2mindate
    else:
        return None, None

def any_negative(birth2maxdate, birth2mindate):
    if birth2maxdate is not None and birth2mindate is not None:
        if birth2maxdate < 0 or birth2mindate < 0:
            return 1
    return 0

def title_list_len(title_list):
    return len(title_list) if isinstance(title_list, list) else 0

def author_length(author):
    return len(author)

# Master function to process each row
def process_row(row):
    birth = row['VIAF_birthdate']
    pubdates_str = row['S2_pubdates']
    author = str(row['author'])
    
    if isinstance(pubdates_str, str) and pubdates_str.strip():
        try:
            pubdates = literal_eval(pubdates_str)
        except (SyntaxError, ValueError):
            pubdates = None
    else:
        pubdates = None

    if isinstance (pubdates, tuple):
        pubdates = list(pubdates)
    if isinstance (pubdates, list):
        pubdates = pubdates
    if pubdates is None:
    # if pubdates is None:
    #if pubdates is None or not isinstance(pubdates, list):

        return {
            'birth2maxdate': None,
            'abs_birth2maxdate': None,
            'birth2mindate': None,
            'abs_birth2mindate': None,
            'negative_status': None,
            'title_length': title_list_len(row['S2_Titlelist']),
            'author_length': author_length(author),
            'S2_pubdates': pubdates,
            'birthyear':birth,
            'S2 titlelist':row['S2_Titlelist'],
            'VIAF_titlelist':row['VIAF_titlelist'],
            'selected_birthyear':row['selected_birthyear'],
            'author':row['author']

    }
    birth2maxdate_value, absbirth2maxdate_value = birth2maxdate(birth, pubdates)
    birth2mindate_value, absbirth2mindate_value = birth2mindate(birth, pubdates)
    neg_status = any_negative(birth2maxdate_value, birth2mindate_value)
    title_length = title_list_len(row['S2_Titlelist'])
    author_len = author_length(row['author'])
    avg_pubdate = find_avg_pubdate(pubdates)

    return {
        'birth2maxdate': birth2maxdate_value,
        'abs_birth2maxdate': absbirth2maxdate_value,
        'birth2mindate': birth2mindate_value,
        'abs_birth2mindate': absbirth2mindate_value,
        'negative_status': neg_status,
        'title_length': title_length,
        'author_length': author_len,
        'S2_pubdates': pubdates,
        'birthyear':birth,
        'S2 titlelist':row['S2_Titlelist'],
        'VIAF_titlelist':row['VIAF_titlelist'],
        'selected_birthyear':row['selected_birthyear'],
        'author':row['author']
        


    }

# Assuming df is your DataFrame
df = result_df.apply(process_row, axis=1, result_type='expand')

# Print or use result_df as needed
# print(result_df)


# In[573]:


df


# In[615]:


df_notnull = df.loc[df['birth2maxdate'].notnull()]


# In[616]:


df_notnull


# In[574]:


import pandas as pd
import ast


# Function to compute average publication date
def compute_avg_pubdate(pubdates):
    try:
        # Convert the string representation to an actual tuple
        pubdates_tuple = ast.literal_eval(pubdates)
        
        # Check if it's a tuple of integers
        if isinstance(pubdates_tuple, tuple) and all(isinstance(date, int) for date in pubdates_tuple):
            return sum(pubdates_tuple) / len(pubdates_tuple)
        else:
            return None
    except (ValueError, SyntaxError):
        return None

# Apply the function to each row
df['avg_pubdate'] = df['S2_pubdates'].apply(lambda x: compute_avg_pubdate(str(x)))



# In[575]:


#publication_age
df['publication_age'] = df['avg_pubdate'] - df['birthyear']


# In[576]:


df


# In[577]:


df['publication_age'] = pd.to_numeric(df['publication_age'], errors='coerce')


# Add status variable, and then convert it from string to numbers

# In[578]:


df['status'] = ""

for idx, row in df.iterrows():
    if row['publication_age'] < 0:
        df.at[idx,'status'] = 'not_born'
    elif row['publication_age'] > 100:
        df.at[idx,'status'] = 'zombie'
    elif row['publication_age'] < 9:
        df.at[idx,'status'] = 'toddler'
    else:
        df.at[idx,'status'] = '0'



# In[579]:


df.loc[df['status'] != '0']


# In[580]:


df['status'] = df['status'].str.replace('zombie', '1')
df['status'] = df['status'].fillna('0')
df['status'] = df['status'].str.replace('not_born', '2')
df['status'] = df['status'].str.replace('toddler', '3')


# In[581]:


# df.loc[df['status'] == '2']


# Now S2 and VIAF titlelist embeddings

# In[582]:


#embeddings

from sentence_transformers import SentenceTransformer

# Load the pre-trained model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings for a list of titles
def get_embeddings(titles_list):
    if isinstance(titles_list, str):
        return model.encode(titles_list)
    else:
        return model.encode('')

# Generate embeddings for each row in the DataFrame
df['S2_embeddings'] = df['S2 titlelist'].apply(get_embeddings)
df['VIAF_embeddings'] = df['VIAF_titlelist'].apply(get_embeddings)


# In[583]:


df['S2_titlelist'] = df['S2 titlelist']


# In[584]:


# #parse embeddings
# #fix embedding issue like we determined in earlier notebook:
# import numpy as np
# import ast

# def parse_embedding(embedding_str):
#     try:
#         # Remove leading and trailing brackets
#         # embedding_str = embedding_str.strip('[]')
        
#         # Replace newlines with spaces and split by spaces
#         # embedding_list = embedding_str.replace('\n', ' ').split()
        
#         # Convert list to numpy array
#         return np.array([float(x) for x in embedding_list], dtype=np.float32)
#     except ValueError as e:
#         print(f"Error parsing embedding: {e}")
#         return np.array([])  # return an empty array in case of error


# df['S2_embeddings_array'] = df['S2_embeddings'].apply(parse_embedding)
# df['VIAF_embeddings_array'] = df['VIAF_embeddings'].apply(parse_embedding)


#I think this step isn't needed since the embeddings WERE created within this notebook



# In[585]:


df['S2_embeddings'] = df['S2_embeddings'].apply(np.mean)
df['VIAF_embeddings'] = df['VIAF_embeddings'].apply(np.mean)


# In[586]:


#get overlapping words and overlapping lemmas:
# Function to lemmatize text and return a set of lemmatized words
nlp = spacy.load("en_core_web_sm")

def lemmatize_text(text):
    doc = nlp(text)
    return set(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

# Function to calculate lemma overlap and return the overlapping lemmas
def calculate_lemma_overlap(text1, text2):
    text1 = str(text1)
    text2 = str(text2)
    lemmas1 = lemmatize_text(text1)
    lemmas2 = lemmatize_text(text2)
    overlap = lemmas1.intersection(lemmas2)
    return len(overlap), overlap

# # Apply the function to compute lemma overlap for each row in the DataFrame
# df[['lemma_overlap', 'overlapping_lemmas']] = df.apply(
#     lambda row: pd.Series(calculate_lemma_overlap(row['VIAF_titlelist'], row['S2_titlelist'])), 
#     axis=1)
df[['lemma_overlap', 'overlapping_lemmas']] = df.apply(
    lambda row: pd.Series(calculate_lemma_overlap(row['VIAF_titlelist'], row['S2_titlelist'])), 
    axis=1)


# In[587]:


#add overlaps

#add word overlap as a new feature
import pandas as pd
from nltk.corpus import stopwords
import nltk

# Download stopwords from NLTK if you haven't already
nltk.download('stopwords')
# Define stop words
stop_words = set(stopwords.words('english'))

# Function to create bag of words after removing stop words
def bag_of_words(title):
    words = title.lower().split()
    return set([word for word in words if word not in stop_words])

# Function to find and count word overlap
def find_word_overlap(row):
    try:
        v_list = ast.literal_eval(row['VIAF_titlelist'])
        s_list = ast.literal_eval(row['S2_titlelist'])
        if isinstance(v_list, list):

            for title in v_list:
                title = str(title)
        if isinstance(s_list, list):

            for title in s_list:
                title = str(title)
        if isinstance(v_list, list):

            v_bag = bag_of_words(v_list) 
        if isinstance(s_list, list):

            s_bag = bag_of_words(s_list) 

        overlap = v_bag & s_bag
    except:
        overlap = ''
    return pd.Series([len(overlap), list(overlap)])
  
# Apply the function to each row and create two new columns
df[['word_overlap_count', 'overlapping_words']] = df.apply(find_word_overlap, axis=1)


# In[588]:


#embed overlapping word meaning
import spacy
import numpy as np

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

def get_word_embeddings(words):
    embeddings = [nlp(word).vector for word in words if nlp(word).has_vector]
    if not embeddings:  # Handle case where no valid embeddings
        return np.zeros(nlp.vocab.vectors.shape[1])
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding


df['mean_embedding'] = df['overlapping_words'].apply(get_word_embeddings)


# In[589]:


#rename to keep track of where birthdate data came from
df['VIAF_birthdate'] = df['birthyear']
df['pub_age'] = df['publication_age']


# In[590]:


#store the metadata to examine later with the probabilities
original_indices = df.index
original_metadata = df[['VIAF_titlelist', 'selected_birthyear', 'author', 'S2_titlelist','status','pub_age','avg_pubdate', 'VIAF_birthdate','overlapping_words','word_overlap_count','lemma_overlap','overlapping_lemmas']].copy()




# In[591]:


df['status'] = df['status'].apply(pd.to_numeric)


# In[592]:


df


# In[593]:


df_2


# In[594]:


#in this case, I had done the true labels by hand, so let's add them in here
df_2 = pd.read_csv('new_labels_aug22.csv')
df['match?'] = df_2['new_label']


# In[595]:


import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                             confusion_matrix, ConfusionMatrixDisplay,
                             classification_report, accuracy_score)
import matplotlib.pyplot as plt


# X = df.drop(columns=['author','VIAF_titlelist','S2_titlelist','overlapping_words','selected_birthyear','overlapping_lemmas','mean_embedding','matched_title?','match','match?','title_list','record_enumerated_titles','S2titles','matched_title_list','normalized_author','common_words','S2_Author','notes','S2_pubdates','S2_Titlelist','selected_birthyear'])  # Drop the label and metadata columns

X = df.drop(columns=['author','VIAF_titlelist','S2_titlelist','overlapping_words','selected_birthyear','overlapping_lemmas','mean_embedding','match?','S2_pubdates','selected_birthyear','negative_status', 'S2 titlelist'])
y = df['match?']

# Optional: If you want to split into training and test sets first
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# In[596]:


# df[['birthdate','standard_birthdate','VIAF_birthdate']]


# In[597]:


# For DataFrames and Series, you can also check the data types of columns
print("\nData types in X_train:")
print(X_train.dtypes)
print("\nData types in y_train:")
print(y_train.dtype)


# In[598]:


X['birth2mindate'] = pd.to_numeric(X['birth2mindate'], errors= 'coerce')
X['birth2maxdate'] = pd.to_numeric(X['birth2maxdate'], errors= 'coerce')

X['abs_birth2mindate'] = pd.to_numeric(X['abs_birth2mindate'], errors= 'coerce')

X['abs_birth2maxdate'] = pd.to_numeric(X['abs_birth2maxdate'], errors= 'coerce')


# In[599]:


pd.to_numeric(X['birth2mindate'], errors= 'coerce')


# In[600]:


pd.to_numeric(X['birth2mindate'], errors= 'coerce')


# In[601]:


print(X)


# In[602]:


# For DataFrames and Series, you can also check the data types of columns
print("\nData types in X:")
print(X.dtypes)
print("\nData types in y:")
print(y.dtype)


# In[603]:


#filter methods of feature selection
#correlations
corr_matrix = X.corr()  # Default is Pearson correlation




# In[604]:


plt.figure(figsize=(10, 8))  # Set the figure size
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.show()


# In[605]:


# #coefficents
# print(model.intercept_,model.coef_)


# In[606]:


type(X)


# In[607]:


X.dtypes


# In[610]:


#threshold variance
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

# Convert DataFrame to NumPy array
# X = X.values
# Initialize VarianceThreshold, threshold can be customized (e.g., 0.1 or 0.5)
threshold = 0.1
selector = VarianceThreshold(threshold=threshold)

# Apply the variance threshold
X_reduced = selector.fit_transform(X)

# Get the names of the remaining features
remaining_features = X.columns[selector.get_support()]

# Convert the reduced data back to a DataFrame with the selected features
df_reduced = pd.DataFrame(X_reduced, columns=remaining_features)

print("Original Features:\n", X)
print("\nSelected Features (with variance threshold = {}):\n".format(threshold), df_reduced)


# In[470]:


# Initialize the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')

# Perform cross-validation
# model = RandomForestClassifier()


# In[546]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.datasets import make_regression

# Create a sample dataset (or load your dataset)
X, y = make_regression(n_samples=100, n_features=10, noise=0.1)
df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
df['Target'] = y

# Forward Selection Function
def forward_selection(data, target):
    initial_features = []
    best_features = initial_features.copy()
    while True:
        remaining_features = list(set(data.columns) - set(best_features))
        best_pval = float('inf')
        best_feature = None
        
        for feature in remaining_features:
            model = sm.OLS(data[target], sm.add_constant(data[best_features + [feature]])).fit()
            pval = model.pvalues[feature]
            if pval < best_pval:
                best_pval = pval
                best_feature = feature
                
        if best_pval < 0.05:  # Adjust p-value threshold as needed
            best_features.append(best_feature)
        else:
            break
            
    return best_features

# Backward Selection Function
def backward_selection(data, target):
    features = data.columns.tolist()
    while True:
        model = sm.OLS(data[target], sm.add_constant(data[features])).fit()
        pvals = model.pvalues.iloc[1:]  # Exclude the intercept
        worst_pval = pvals.max()
        
        if worst_pval >= 0.05:  # Adjust p-value threshold as needed
            worst_feature = pvals.idxmax()
            features.remove(worst_feature)
        else:
            break
            
    return features

# Applying Forward Selection
forward_selected_features = forward_selection(df.drop(columns='Target'), 'Target')
print("Forward Selected Features:", forward_selected_features)

# Applying Backward Selection
backward_selected_features = backward_selection(df.drop(columns='Target'), 'Target')
print("Backward Selected Features:", backward_selected_features)


# In[ ]:


#forward and backwards selectio nmethods 


# In[471]:


y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')

# Store predictions with original indices
predictions_df = pd.DataFrame(y_pred_proba, index=original_indices, columns=['Class_0_Proba', 'Class_1_Proba'])

# Merge the metadata back
results_df = pd.concat([original_metadata, predictions_df], axis=1)
results_df['True_Label'] = y.loc[results_df.index]

# Now you have a DataFrame with metadata and predictions
print(results_df.head())


# In[472]:


# y_pred_proba = cross_val_predict(model, X_train, y_train, cv=5, method='predict_proba')
# 
# # Store predictions with original indices
# predictions_df = pd.DataFrame(y_pred_proba, index=original_indices, columns=['Class_0_Proba', 'Class_1_Proba'])
# 
# # Merge the metadata back
# results_df = pd.concat([original_metadata, predictions_df], axis=1)
# results_df['True_Label'] = y.loc[results_df.index]
# 
# # Now you have a DataFrame with metadata and predictions
# print(results_df.head())


# In[395]:


# #including the testing and training sets
# 
# predictions_df = pd.DataFrame(y_pred_proba, index=X_train.index, columns=['Class_0_Proba', 'Class_1_Proba'])
# 
# # Merge the metadata with predictions using original indices
# results_df = pd.concat([original_metadata.loc[X_train.index], predictions_df], axis=1)
# 
# # Add the true labels to the DataFrame
# results_df['True_Label'] = y_train
# 
# # Display the first few rows of the DataFrame
# print(results_df.head())


# In[396]:


# from sklearn.metrics import classification_report
# 
# # Convert predicted probabilities to class labels (0 or 1 based on threshold 0.5)
# y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)
# 
# 
# # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
# report = classification_report(y_train, y_pred)
# 
# # Print the classification report
# print(report)


# In[397]:


#all data included

y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')

# Store predictions with original indices
predictions_df = pd.DataFrame(y_pred_proba, index=original_indices, columns=['Class_0_Proba', 'Class_1_Proba'])

# Merge the metadata back
results_df = pd.concat([original_metadata, predictions_df], axis=1)
results_df['True_Label'] = y.loc[results_df.index]

# Now you have a DataFrame with metadata and predictions
print(results_df.head())


# In[400]:


from sklearn.metrics import classification_report

# Convert predicted probabilities to class labels (0 or 1 based on threshold 0.5)
# y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

y_pred = (y_pred_proba[:, 1] >= 0.8).astype(int)


# Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
report = classification_report(y, y_pred)

# Print the classification report
print(report)


# In[401]:


y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)


# Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
report = classification_report(y, y_pred)

# Print the classification report
print(report)


# In[404]:


y_pred = (y_pred_proba[:, 1] >= 0.7).astype(int)


# Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
report = classification_report(y, y_pred)

# Print the classification report
print(report)


# In[403]:


# y_pred = (y_pred_proba[:, 1] >= 0.9).astype(int)
# 
# 
# # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
# report = classification_report(y, y_pred)
# 
# # Print the classification report
# print(report)


# In[399]:


#results
# 
# from sklearn.metrics import confusion_matrix
# from sklearn.metrics import classification_report
# confusion_matrix = confusion_matrix(y,y_pred)
# print(confusion_matrix) #gives true pos, false neg etc
# 
# classification_report = classification_report(y, y_pred) #gives precision, f1_score etc
# print(classification_report)


# In[405]:


#save the selected model to a pickle file
# Save the trained model to a pickle file
with open('viaf_classifier_sept23.pkl', 'wb') as file:
    pickle.dump(model, file)

