import re
import pandas as pd
import pickle
import datetime
import json


# #get the VIAF entries for all data which were processed in another jupyter notebook

import pandas as pd


#then we want to run a very similar process to get features as we did when we created the model originally
# %%
# imports
import re
import requests
from ast import literal_eval
from scipy.spatial.distance import cosine

import time
import string
import unicodedata
import pprint

from nltk.corpus import stopwords
import nltk
import spacy

nlp = spacy.load("en_core_web_sm")

# Download stopwords from NLTK if you haven't already
nltk.download('stopwords')
# Define stop words
stop_words = set(stopwords.words('english'))

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


def update_pubdates(name):
    if is_author_in_dict(name, S2_data_dict):
        years = S2_data_dict[name]
    else:
        print('no')
        years = 'None'
    return years


def get_name(name):
    params = {
        'query': f'local.personalNames = "{name}"',
        'maximumRecords': 10,
        'startRecord': 1,
        'sortKeys': 'holdingscount',
        'httpAccept': 'application/json'
    }

    headers = {'User-Agent': user_agent}
    url = "https://viaf.org/viaf/search"

    r = requests.get(url, params=params, headers=headers, timeout=10)
    data = r.json()

    return data


import pandas as pd

processing_errors = []


# Function to perform search and return DataFrame with search results
def search_author(author_name):
    # Initialize list to store search results
    search_results = []

    try:
        data = get_name(author_name)
        records = data['searchRetrieveResponse']['records']
        total_records = len(records)

        for idx, record in enumerate(records):
            record_data = record['record']['recordData']
            birthdate = record['record']['recordData']['birthDate']
            viaf_title_list = []
            # Extract titles if available
            if 'titles' in record_data:
                titles_data = record_data['titles']
                if titles_data is not None:
                    if isinstance(titles_data['work'], list):
                        # Extract titles from the list of works
                        viaf_title_list.extend([work['title'] for work in titles_data['work'] if 'title' in work])
                    else:
                        # Extract title from a single work
                        title = titles_data['work'].get('title')
                        if title:
                            viaf_title_list.append(title)


            # title_list = df.loc[df['author'] == author_name, 'title_list'].iloc[0]

            search_results.append({'author': author_name, 'record_count': len(records),
                                   'record_enumerated': idx, 'viaf_title_list': viaf_title_list,
                                   'birthdate': birthdate})
    except Exception as e:
        print(f'Processing Error for {author_name}: {e}')
        processing_errors.append(author_name)

    return search_results


user_agent = 'YOUR PROJECT NAME HERE'
id_column_name = "Name"

pause_between_req = 1

use_title_reconcilation = True

cache = {}


def find_max_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return max(pubdates)
    return None


def find_min_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return min(pubdates)
    return None


def find_avg_pubdate(pubdates):
        if isinstance(pubdates, int):
            avg_pubdate = pubdates
            return avg_pubdate
        if isinstance(pubdates, float):
            pubdates = str(pubdates)
            if len(pubdates) > 5:
                pubdates = pubdates[:4]
                pubdates = int(pubdates)
                avg_pubdate = pubdates
                return avg_pubdate
        if isinstance(pubdates, str) and pubdates != 'no date' and pubdates != "" and  pubdates != 'nan':
            # pubdates = pubdates.strip('-')
            pubdates = pubdates.replace('-', '')
            if len(pubdates) > 5:
                pubdates = pubdates[:4]
                try:
                    pubdates = int(pubdates)
                except ValueError as e:
                    print(f"Error converting pubdates to int: {e}")
                    avg_pubdate = 'error'
                    return avg_pubdate
            avg_pubdate = pubdates
            return avg_pubdate
        if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
            avg_pubdate =  (sum(pubdates) / len(pubdates))
        else:
            avg_pubdate = 0
        return avg_pubdate





weird_cases_to_examine = []
def birth2maxdate(birth, pubdates,author):
    # Convert the string representation to an actual tuple
    # pubdates_tuple = literal_eval(pubdates)

    # Check if it's a tuple of integers
    # if isinstance(pubdates_tuple, tuple) and all(isinstance(date, int) for date in pubdates_tuple):
    if isinstance(birth, str):
        if len(birth) >= 8:
            birth = birth[:4]
    birth = int(birth)
    if isinstance(pubdates, list) or isinstance(pubdates, tuple):
        if len(pubdates) > 1:
            max_pubdate = find_max_pubdate(pubdates)
            if max_pubdate is not None:
                birth2maxdate = max_pubdate - birth
            abs_birth2maxdate = abs(birth2maxdate)
            return birth2maxdate, abs_birth2maxdate

        elif len(pubdates) == 1:
            # if pubdates == '"' or pubdates == '':
            if author in unique_authors_to_search:
                if author in S2_data_dict.keys():
                    pubdates  = S2_data_dict[author]['year']
                    max_pubdate = find_max_pubdate(pubdates)
                    birth2maxdate = max_pubdate - birth
                    abs_birth2maxdate = abs(birth2maxdate)

                    return birth2maxdate, abs_birth2maxdate
                else:
                    weird_cases_to_examine.append(author)
                    return '',''


            # else:
            #     max_pubdate = int(pubdates[0])
            #     birth2maxdate = max_pubdate - birth
            #     abs_birth2maxdate = abs(birth2maxdate)
    if isinstance(pubdates, int):
        max_pubdate = pubdates
        birth2maxdate = max_pubdate - birth
        abs_birth2maxdate = abs(birth2maxdate)
        try:
            return birth2maxdate, abs_birth2maxdate
        except:
            return None, None




def birth2mindate(birth, pubdates,author):
    # Convert the string representation to an actual tuple
    # pubdates_tuple = literal_eval(pubdates)

    # Check if it's a tuple of integers
    # if isinstance(pubdates_tuple, tuple) and all(isinstance(date, int) for date in pubdates_tuple):
    if isinstance(birth, str):
        if len(birth) >= 8:
            birth = birth[:4]
    birth = int(birth)
    if isinstance(pubdates, list) or isinstance(pubdates, tuple):
        if len(pubdates) > 1:
            min_pubdate = find_min_pubdate(pubdates)
            if min_pubdate is not None:
                birth2mindate = min_pubdate - birth
            abs_birth2mindate = abs(birth2mindate)
        elif len(pubdates) == 1:
            # if pubdates == '"' or pubdates == '':
            if author in unique_authors_to_search:
                if author in S2_data_dict.keys():
                    pubdates = S2_data_dict[author]['year']
                    min_pubdate = find_min_pubdate(pubdates)
                    birth2mindate = min_pubdate - birth
                else:
                    weird_cases_to_examine.append(author)
                    return None, None
    if isinstance(pubdates, int):
        min_pubdate = pubdates
        birth2mindate = min_pubdate - birth
        abs_birth2mindate = abs(birth2mindate)
    return birth2mindate, abs_birth2mindate

def clean_up_pubdates(pubdates):
    if pubdates is None:
        return None
    for pubdate in pubdates:
            if 'no date' in pubdate:
                pubdates.remove(pubdate)
            if len(pubdates) >= 8:
                pubdates = clean_pubdates(pubdates)
            return pubdates

def any_negative(birth2maxdate, birth2mindate):
    if birth2maxdate is not None and birth2mindate is not None:
        if birth2maxdate < 0 or birth2mindate < 0:
            return 1
    return 0


def title_list_len(title_list):
    return len(title_list) if isinstance(title_list, list) else 0


def author_length(author):
    return len(author)


# Function to generate embeddings for a list of titles
def get_embeddings(titles_list):
    if isinstance(titles_list, str):
        return model.encode(titles_list)
    else:
        return model.encode('')


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


# Function to create bag of words after removing stop words
def bag_of_words(title):
    words = title.lower().split()
    return set([word for word in words if word not in stop_words])


# Function to find and count word overlap
def find_word_overlap(row):
    # try:
    if not pd.isna(row['VIAF_titlelist']) and not pd.isna(row['S2_titlelist']):
        v_list = literal_eval((row['VIAF_titlelist']))
        s_list = literal_eval(str(row['S2_titlelist']))
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
    else:
        overlap = ''
    # except:
    #     overlap = ''
    return pd.Series([len(overlap), list(overlap)])


# Function to create bag of words after removing stop words
def bag_of_words(title):
    words = title.lower().split()
    return set([word for word in words if word not in stop_words])


# Function to find and count word overlap
def find_word_overlap(row):
    if not pd.isna(row['VIAF_titlelist']) and not pd.isna(row['S2_titlelist']):
        # try:
        # Convert string representations to lists using ast.literal_eval
        v_list = literal_eval(row['VIAF_titlelist'])
        s_list = str(row['S2_titlelist'])
        s_list = (s_list).split(',')
        s_list = [item for item in s_list if item != 'nan']

        # Initialize sets for storing bag of words
        v_bag = set()
        s_bag = set()

        # Create bag of words for VIAF titles
        if isinstance(v_list, list):
            for title in v_list:
                title = str(title)
                v_bag.update(bag_of_words(title))  # Update the set with words

        # Create bag of words for S2 titles
        if isinstance(s_list, list):
            for title in s_list:
                title = str(title)
                s_bag.update(bag_of_words(title))  # Update the set with words

        # Find overlap between the two sets of words
        overlap = v_bag & s_bag
    # except (ValueError, SyntaxError):
    #     overlap = set()  # In case of any parsing error
    else:
        overlap = set()  # If either list is missing

    # Return the overlap count and the list of overlapping words
    return pd.Series([len(overlap), list(overlap)])


def get_word_embeddings(words):
    embeddings = [nlp(word).vector for word in words if nlp(word).has_vector]
    if not embeddings:  # Handle case where no valid embeddings
        return np.zeros(nlp.vocab.vectors.shape[1])
    mean_embedding = np.mean(embeddings, axis=0)
    return mean_embedding


def get_cosine_distance_bw_title_embeddings(VIAF_embedding, S2_embedding):
    cosine_dist = cosine(VIAF_embedding, S2_embedding)
    return cosine_dist

def process_pubdates_and_birth(pubdates_str, birth,author):
    pubdates = None  # Initialize pubdates to avoid UnboundLocalError

    # Process pubdates_str based on its type
    if isinstance(pubdates_str, str):
        if len(pubdates_str) > 4:
            if ',' in pubdates_str:
                pubdates = pubdates_str.split(',')

            else:
                pubdates = [pubdates_str]  # Convert string to a single-element list

        else:
            # if pubdates_str != '' and pubdates_str != 'no date' or pubdates_str != 'no d':
            if pubdates_str != '' and pubdates_str not in ['no date', 'no d'] and pubdates_str[:4].isdigit():
                pubdates = int(pubdates_str[:4])  # Extract first 4 characters and convert to int if needed
    #
    # elif isinstance(pubdates_str, (tuple, list)):
    #     pubdates = list(pubdates_str)  # Convert tuple to list, if it's a tuple
    #     if len(pubdates) == 1 and pubdates[0] != 'no date':
    #         pubdates = int(pubdates[0][:4]) if len(pubdates[0]) >= 4 else pubdates[0]
    #     else:
    #         pubdates = pubdates
    #
    # elif isinstance(pubdates_str, int):
    #     pubdates = pubdates_str  # If already an int, no need to process further
    #
    # elif pubdates_str == '':  # Handle empty string case
    #     pubdates = ''
    #
    # # Handle specific cases with special strings
    # if pubdates_str == '"' or pubdates_str == "''" or pubdates_str == '' or pubdates is None:
    #     try:
    #         if author in S2_data_dict.keys():
    #             pubdates = S2_data_dict[author]['year']
    #     except KeyError:
    #         pubdates = 0

    # Process birth date if it's a string
    if isinstance(birth, str):
        birth = birth.replace('-', '')
        if len(birth) > 5:
            birth = int(birth[:4])  # Only take the first 4 characters of the birth year
        else:
            birth = int(birth)

    return pubdates, birth



def process_row(row):
    birth = row['birthdate']
    pubdates_str = str(row['S2_pubdates'])
    if len(pubdates_str)>= 5:
        pubdates_str = pubdates_str[:4]
    if pubdates_str == 'no date' or pubdates_str == '' :
        # if row['S2_Year'] != 'NaN' and row['S2_Year'] != 'nan':
        if pd.notna(row['S2_Year']):
            pubdates_str = (str(row['S2_Year']))
            pubdates_str = pubdates_str[:4]
            pubdates_str = int(pubdates_str)
        else:
            pubdates_str = 0
        if isinstance(birth, str):
            if len(birth) > 4:
                birth = int(birth[:4])
            else:
                birth = int(birth)
    author = str(row['author'])
    pubdates, birth = process_pubdates_and_birth(pubdates_str, birth, author)




    if pubdates is None:



            return {
                    'birth2maxdate': None,
                    'abs_birth2maxdate': None,
                    'birth2mindate': None,
                    'abs_birth2mindate': None,
                    'negative_status': None,
                    'title_count': title_list_len(row['S2_titlelist']),
                    'author_length': author_length(author),
                    'S2_pubdates': None,
                    'VIAF_birthdate': None,
                    'S2 titlelist': row['S2_titlelist'],
                    'VIAF_titlelist': row['VIAF_titlelist'],
                    'author': row['author'],
                    'pub_age': row['publication_age'],
                    'avg_pubdate':row['avg_pubdate']}


    else:
        # if pubdates is not None and birth is not None:
        if pubdates and birth:
            # test = birth2maxdate(birth, pubdates, author)
            # print(test)
            # if pd.notna(birth):
            #     if pd.notna(pubdates):
            # if pd.notna(test):
            birth2maxdate_value, absbirth2maxdate_value = birth2maxdate(birth, pubdates, author)
            birth2mindate_value, absbirth2mindate_value = birth2mindate(birth, pubdates, author)
            neg_status = any_negative(birth2maxdate_value, birth2mindate_value)
            title_count = title_list_len(row['S2_titlelist'])
            author_len = author_length(row['author'])
            avg_pubdate = find_avg_pubdate(pubdates)
            return {
            'birth2maxdate': birth2maxdate_value,
            'abs_birth2maxdate': absbirth2maxdate_value,
            'birth2mindate': birth2mindate_value,
            'abs_birth2mindate': absbirth2mindate_value,
            'negative_status': neg_status,
            'title_count': title_count,
            'author_length': author_len,
            'S2_pubdates': pubdates,
            'VIAF_birthdate': birth,
            'S2 titlelist': row['S2_titlelist'],
            'VIAF_titlelist': row['VIAF_titlelist'],
            'author': row['author'],
            'pub_age': row['publication_age'],
            ' avg_pubdate': row['avg_pubdate']

        }

        elif pubdates:
            return {
                'birth2maxdate': None,
                'abs_birth2maxdate': None,
                'birth2mindate': None,
                'abs_birth2mindate': None,
                'negative_status': None,
                'title_count': title_list_len(row['S2_titlelist']),
                'author_length': author_length(author),
                'S2_pubdates': pubdates,
                'VIAF_birthdate': None,
                'S2 titlelist': row['S2_titlelist'],
                'VIAF_titlelist': row['VIAF_titlelist'],
                'author': row['author'],
                'pub_age': row['publication_age'],
                'avg_pubdate': row['avg_pubdate']}


    #         else:
    #             print('Test was NaN')

def jaccard_distance_for_lists(list1, list2):
    # Convert both lists into sets (to remove duplicates within each list)
    set1 = set(list1)
    set2 = set(list2)

    # Calculate the intersection and union
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))

    # If both sets are empty, return 0 (no distance)
    if union == 0:
        return 0

    # Jaccard similarity
    jaccard_sim = intersection / union

    # Jaccard distance is 1 - Jaccard similarity
    return 1 - jaccard_sim


def apply_jaccard(df, col1, col2):
    # Apply the Jaccard distance function row-wise
    df['jaccard_distance'] = df.apply(lambda row: jaccard_distance_for_lists(row[col1], row[col2]), axis=1)
    return df
def clean_pubdates(pubdates):
    # If the pubdates are passed as a string that looks like a list, evaluate it
    if isinstance(pubdates, str):
        pubdates = literal_eval(pubdates)

    # If the pubdates is a list, process each date
    if isinstance(pubdates, list):
        for i in range(len(pubdates)):
            pubdate = pubdates[i]

            # Handle YYYYMMDD format
            if len(pubdate) == 8 and pubdate.isdigit():
                pubdates[i] = pubdate[:4]  # Extract the year
            else:
                # Handle other date formats like YYYY-MM-DD
                try:
                    parsed_date = datetime.strptime(pubdate, "%Y-%m-%d")
                    pubdates[i] = str(parsed_date.year)  # Extract the year
                except ValueError:
                    pubdates[i] = pubdate  # Handle invalid formats
    return pubdates

def find_exact_matches_for_author(row):
    matches = []

    # Iterate over each row in the DataFrame
    # for idx, row in df.iterrows():
    author = row['author']

    # Get titles from viaf_title_list and title_list
    viaf_titles = row['VIAF_titlelist'] if isinstance(row['VIAF_titlelist'], list) else []
    title_list = row['S2 titlelist'] if isinstance(row['S2 titlelist'], list) else [row['S2 titlelist']]

    # Convert both lists to lowercase to make the comparison case-insensitive
    viaf_titles_lower = set([str(title).lower() for title in viaf_titles])
    title_list_lower = set([str(title).lower() for title in title_list])

    # Find the intersection of titles between viaf_titles and title_list
    common_titles = viaf_titles_lower.intersection(title_list_lower)

    # If there are matching titles, add them to the results
    if common_titles:
        matches.append({
            'author': author,
            'matching_titles': list(common_titles)
        })

    return matches
def count_matches(exact_title_match):
    return len(exact_title_match)
def extract_year(df, column_name):
    # Apply a function to each row of the specified column
    df[column_name] = df[column_name].apply(lambda x: int(x[:4]) if isinstance(x, str) and len(x) > 8 else x)
    return df

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)





if __name__ == '__main__':

    file_path = 'LitMetadataWithS2 (3).tsv'
    meta = pd.read_csv(file_path, sep='\t')
    meta['author'] = meta['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Use the explode method to expand the authors column
    meta_exploded = meta.explode('authors')
    meta = meta_exploded


    meta['author'] = meta['author'].astype(str).apply(normalize_text)

    # test = meta.loc[meta['author'] == 'h t betteridge']
    # print(test.head())


    import pandas as pd
    import json


    df = pd.read_csv('../result_df_july18_1136am.csv')


    print(df.head(30))

    print(df['author'].head(30))

    print(df.columns)

    # columns_to_drop = ['Unnamed: 0', 'index', 'birthdate', 'record_enumerated','title_list','record_count']
    # columns_to_drop = ['Unnamed: 0', 'index', 'birthdate', 'record_enumerated','title_list','record_count']
    # df = df[:100]

    # df = df.drop(columns_to_drop)

    for idx, row in df.iterrows():
        author = str(row['author'])
        author_clean = normalize_text(author)
        df.at[idx, 'author'] = author_clean



    unique_author_names = []

    for author_name in df['author']:

        if author_name not in unique_author_names:
            unique_author_names.append(author_name)




        # create rows?
    from ast import literal_eval

    # new_rows = []
    # for author_entry in unique_author_names:
    #     author_entry = normalize_text(author_entry)
    #     # author = literal_eval(author)
    #     if ',' in author_entry:
    #         author_list = author_entry.split(',')
    #         # author_list = literal_eval(author_list)
    #         for author in author_list:
    #             author = normalize_text(author)
    #     else:
    #         author = author_entry
    #     if author != 'nan' and author != "''":
    #         # Filter rows where the author matches
    #         # for i in meta['author']:
    #         #     if meta['author'][i] == author:
    #         filtered_meta = meta[meta['author'] == author]
    #         # if filtered_meta.empty:
    #         #     continue
    #         # else:
    #         # Create new rows as a list of dictionaries
    #         # rows = filtered_meta.apply(lambda row: {
    #         #     'author': row['author'],
    #         #     'journal': row['journal'],
    #         #     'year': row['year'],
    #         #     'title': row['title'],
    #         #     'S2titles': row['S2titles'],
    #         #     'S2Years': row['S2years'],
    #         #     'S2Year': row['year']
    #         # }, axis=1).tolist()
    #
    #         # Create the new DataFrame by copying relevant columns
    #         new_row = filtered_meta[['author', 'journal', 'year', 'title', 'S2titles', 'S2years']].copy()
    #         new_rows.append(new_row)

    author_dict = {}  # This will store the final result with authors as keys

    # Loop over each unique author entry
    for author_entry in unique_author_names:
        author_entry = normalize_text(author_entry)  # Normalize the author entry

        # If there are multiple authors in the entry, split by comma
        if ',' in author_entry:
            author_list = author_entry.split(',')  # Split by comma
            author_list = [normalize_text(author) for author in author_list]  # Normalize each author
        else:
            author_list = [author_entry]  # Treat a single author as a list of one

        # Loop over each individual author
        for author in author_list:
            # Skip invalid or placeholder values
            if author != 'nan' and author != "''":
                # Filter rows where the author matches
                filtered_meta = meta[meta['author'] == author]

                # If there are matching rows, add them to the dictionary
                if not filtered_meta.empty:
                    # Create a list of dictionaries for the rows
                    row_list = filtered_meta.apply(lambda row: {
                        'journal': row['journal'],
                        'year': row['year'],
                        'title': row['title'],
                        'S2titles': row['S2titles'],
                        'S2Years': row['S2years'],
                        'S2Year': row['year']
                    }, axis=1).tolist()

                    # If the author is already in the dictionary, extend the list; otherwise, create a new entry
                    if author in author_dict:
                        author_dict[author].extend(row_list)
                    else:
                        author_dict[author] = row_list

    # Now author_dict contains the author names as keys and lists of row data as values
    # print(author_dict)

    # new_row = {
            #             'author': author,
            #             'journal': row['journal'],
            #             'year': row['year'],
            #             'title': row['title'],
            #             'S2titles': row['S2titles'],
            #             'S2Years': row['S2years'],
            #             'S2Year' :row['year']
            #
            #         }
            # rows.append(new_row)



    # Step 1: Create a dictionary mapping author names to their rows
    #         rows_dict = {row['author']: row for row in new_rows}

    # # Step 2: Iterate through the unique author names and retrieve the corresponding row from the dictionary
    # S2_data_dict = {author_name: rows_dict[author_name] for author_name in unique_author_names if
    #                 author_name in rows_dict}

    df['author'] = df['author'].apply(normalize_text)


    def is_author_in_dict(author, dict):
        if author in dict.keys():
            return True


    df['S2_pubdates'] = ""
    unique_authors_to_search = []
    authors_data = {}

    # %%
    import pandas as pd
    import os

    # Path to save the results
    results_file = '../search_results_random_sample.csv'

    # Load existing results if the file exists
    if os.path.exists(results_file):
        all_search_results_df = pd.read_csv(results_file)
    else:
        all_search_results_df = pd.DataFrame()

        # Get the unique author names, skipping those already processed
    if not all_search_results_df.empty:
        processed_authors = all_search_results_df['author'].unique()
        unique_author_names = [name for name in unique_author_names if name not in processed_authors]

    # Chunk processing parameters
    chunk_size = 2000  # Adjust the size as needed
    total_authors = len(unique_author_names)
    print(total_authors)

    # Process in chunks
    for i in range(0, total_authors, chunk_size):
        chunk = unique_author_names[i:i + chunk_size]
        all_search_results = []

        for author_name in chunk:
            search_results = search_author(author_name)
            all_search_results.extend(search_results)

        # Convert list of dictionaries to DataFrame
        chunk_df = pd.DataFrame(all_search_results)

        # Append to existing results
        all_search_results_df = pd.concat([all_search_results_df, chunk_df], ignore_index=True)

        # Save the updated results to the file
        all_search_results_df.to_csv(results_file, index=False)
        author_missing_from_S2 = []

        all_search_results_df['S2_titlelist'] = ""
        all_search_results_df['S2_pubdates'] = ""


        print(f"Processed authors {i + 1} to {min(i + chunk_size, total_authors)}")
        for idx, row in all_search_results_df.iterrows():
            author = normalize_text(row['author'])
            if author:
                if author in author_dict:
                    S2titles = [entry['S2titles'] for entry in author_dict[author]]  # List of S2titles
                    all_search_results_df.at[idx, 'S2_titlelist'] = S2titles
                    pubdates = [entry['S2Years'] for entry in author_dict[author]]
                    all_search_results_df.at[idx, 'S2_pubdates'] = pubdates
                    pubdates2 = [entry['year'] for entry in author_dict[author]]
                    all_search_results_df.at[idx, 'S2_Year'] = str(pubdates2)
                else:
                    author_missing_from_S2.append(author)

        # # Open the file in write mode
        # with open('S2_data_dict.txt', 'w') as json_file:
        #     json.dump(S2_data_dict, json_file, indent=4)  # indent for pretty printing


    all_search_results_df.to_csv('random_sample_search_results_VIAF_S2_Oct.csv')


import os
# os.system('say "Your script has finished"')