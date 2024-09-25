#%%
#imports
import re
import requests
from ast import literal_eval
from scipy.spatial.distance import cosine

import time
import string
import unicodedata
import pprint

import pandas as pd
#%%


# Display the exploded DataFrame
# print(meta_exploded)

#%%

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
    if name in swapped_dict:
        years = swapped_dict[name]
        return years
    



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
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return (sum(pubdates)/len(pubdates))
    return None

def birth2maxdate(birth, pubdates):
        # Convert the string representation to an actual tuple
        # pubdates_tuple = literal_eval(pubdates)
        
        # Check if it's a tuple of integers
        # if isinstance(pubdates_tuple, tuple) and all(isinstance(date, int) for date in pubdates_tuple):
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
    
    # if isinstance(pubdates_str, str) and pubdates_str.strip():
    if isinstance(pubdates_str, str):
        try:
            pubdates = literal_eval(pubdates_str)
            if isinstance(pubdates, tuple):
                pubdates = list(pubdates)
        except (SyntaxError, ValueError):
            print(f"Error parsing pubdates_str: {pubdates_str}, Error: {e}")
            pubdates = None
  
    if isinstance(pubdates_str, tuple):
        pubdates = list(pubdates_str)
    elif isinstance(pubdates_str, list):
        pubdates = pubdates_str
    # if pubdates is None:
    #     pubdates = None
    if pubdates_str is None:
        pubdates = None
    
    # if pubdates is None:
    #if pubdates is None or not isinstance(pubdates, list):

        return {
            'birth2maxdate': None,
            'abs_birth2maxdate': None,
            'birth2mindate': None,
            'abs_birth2mindate': None,
            'negative_status': None,
            'title_count': title_list_len(row['S2_Titlelist']),
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
    title_count = title_list_len(row['S2_Titlelist'])
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
        'birthyear':birth,
        'S2 titlelist':row['S2_Titlelist'],
        'VIAF_titlelist':row['VIAF_titlelist'],
        'selected_birthyear':row['selected_birthyear'],
        'author':row['author']
        


    }

# Function to compute average publication date
def compute_avg_pubdate(pubdates):
    if pubdates is None:
        return None
    # try:
        # Convert the string representation to an actual tuple
    pubdates_tuple = tuple(pubdates)
    if pubdates_tuple is None:
        return None
    # Check if it's a tuple of integers
    elif isinstance(pubdates_tuple, tuple) and all(isinstance(date, int) for date in pubdates_tuple):
        return sum(pubdates_tuple) / len(pubdates_tuple)
    else:
            for date in pubdates_tuple:
                date = int(date)
                pubdates_tuple_2.add(date)
                return sum(pubdates_tuple) / len(pubdates_tuple)
    # else:
    #     return None
    # # except (ValueError, SyntaxError):
    #     return None



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


####
if __name__ == '__main__':
    df = pd.read_csv('result_df_july18_1136am.csv')
    file_path = 'LitMetadataWithS2 (3).tsv'
    meta = pd.read_csv(file_path, sep='\t')
    # %%
    # Ensure the authors column is of type list
    meta['author'] = meta['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)

    # Use the explode method to expand the authors column
    meta_exploded = meta.explode('authors')

    meta = meta_exploded

    for idx, row in meta.iterrows():
        author = str(row['author'])
        author_clean = normalize_text(author)
        meta.at[idx, 'author'] = author_clean
    # Add back the S2 titles
    # %%
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

    # %%
    title_author_dict = author_titles

    df['normalized_author'] = df['author'].apply(normalize_text)

    # %%
    normalized_dict = {normalize_text(str(author)): normalize_text(str(title)) for title, author in author_titles.items()}

    # %%
    swapped_dict = {v: k for k, v in normalized_dict.items()}

    # %%
    # swapped_dict
    # %%
    df['S2_Titlelist'] = df['normalized_author'].map(swapped_dict)

    # %%
    # df
    # %% md
    # Now add back the S2 pubyears
    # %%
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

    # %%
    # normalized_dict = {normalize_text(str(author)): normalize_text(str(title)) for title, author in title_author_dict.items()}
    normalized_dict = {normalize_text(str(author)): title for title, author in author_years.items()}

    # %%
    # normalize then apply dictionary
    # cleaned_dict = {key.strip("[]'"): value for key, value in normalized_dict.items()}
    swapped_dict = {v: k for k, v in normalized_dict.items()}

    df['S2_pubdates'] = df['normalized_author'].apply(update_pubdates)
    #%%
    # df['S2_pubdates'] = df['author'].map(author_years)
    #%%
    df['S2_pubdates']

    # print(all_search_results_df)

    #%% md
    # Rename some columns so they match later usage
    #%%
    result_df = df
    result_df['VIAF_titlelist'] = result_df['record_enumerated_titles']
    result_df['S2_titlelist'] = result_df['S2titles']
    result_df['average_S2_pubdate'] = result_df['avg_pubdates']
    result_df['VIAF_birthdate'] = result_df['standard_birthdate']
    result_df['S2_Author'] = result_df['author']


    df = result_df
    #%%
    df

    # Assuming df is your DataFrame
    df = result_df.apply(process_row, axis=1, result_type='expand')

    # Print or use result_df as needed
    # print(result_df)

    #%%
    df
    #%%
    df_notnull = df.loc[df['birth2maxdate'].notnull()]
    #%%
    df_notnull
    #%%

    #%%
    import pandas as pd
    import ast

    pubdates_tuple_2 = {}

    df['avg_pubdate'] = df['S2_pubdates'].apply(lambda x: compute_avg_pubdate(x))


    #%%
    df_notnull = df.loc[df['avg_pubdate'].notnull()]
    #%%
    df_notnull
    #%%
    #publication_age
    df['publication_age'] = df['avg_pubdate'] - df['birthyear']

    #%%
    df
    #%%
    df['publication_age'] = pd.to_numeric(df['publication_age'], errors='coerce')

    #%% md
    # Add status variable, and then convert it from string to numbers
    #%%
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


    #%%
    df.loc[df['status'] != '0']
    #%%
    df['status'] = df['status'].str.replace('zombie', '1')
    df['status'] = df['status'].fillna('0')
    df['status'] = df['status'].str.replace('not_born', '2')
    df['status'] = df['status'].str.replace('toddler', '3')

    #%%
    # df.loc[df['status'] == '2']
    #%% md
    # Now S2 and VIAF titlelist embeddings
    #%%
    #embeddings

    from sentence_transformers import SentenceTransformer

    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for each row in the DataFrame
    df['S2_embeddings'] = df['S2 titlelist'].apply(get_embeddings)
    df['VIAF_embeddings'] = df['VIAF_titlelist'].apply(get_embeddings)
    #%%
    df['S2_titlelist'] = df['S2 titlelist']
    #%%

    #%%
    # import numpy as np
    # df['S2_embeddings'] = df['S2_embeddings'].apply(np.mean)
    # df['VIAF_embeddings'] = df['VIAF_embeddings'].apply(np.mean)
    #%%
    #get overlapping words and overlapping lemmas:
    # Function to lemmatize text and return a set of lemmatized words
    import spacy
    nlp = spacy.load("en_core_web_sm")

    df[['lemma_overlap', 'overlapping_lemmas']] = df.apply(
        lambda row: pd.Series(calculate_lemma_overlap(row['VIAF_titlelist'], row['S2_titlelist'])),
        axis=1)

    #%%
    #add overlaps

    #add word overlap as a new feature
    import pandas as pd
    from nltk.corpus import stopwords
    import nltk

    # Download stopwords from NLTK if you haven't already
    nltk.download('stopwords')
    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Apply the function to each row and create two new columns
    df[['word_overlap_count', 'overlapping_words']] = df.apply(find_word_overlap, axis=1)


    #%%

    df['cosine_distance'] = ""

    for idx, row in df.iterrows():
        VIAF_embedding = row['VIAF_embeddings']
        S2_embedding = row['S2_embeddings']
        cosine_dist = get_cosine_distance_bw_title_embeddings(S2_embedding, VIAF_embedding)
        df.at[idx, 'cosine_distance'] = cosine_dist
    #%%
    #embed overlapping word meaning
    import spacy
    import numpy as np

    # Load the spaCy model
    nlp = spacy.load("en_core_web_sm")


    df['mean_embedding'] = df['overlapping_words'].apply(get_word_embeddings)

    #%%
    #rename to keep track of where birthdate data came from
    df['VIAF_birthdate'] = df['birthyear']
    df['pub_age'] = df['publication_age']
    #%%
    #store the metadata to examine later with the probabilities
    original_indices = df.index
    original_metadata = df[['VIAF_titlelist', 'selected_birthyear', 'author', 'S2_titlelist','status','pub_age','avg_pubdate', 'VIAF_birthdate','overlapping_words','word_overlap_count','lemma_overlap','overlapping_lemmas']].copy()



    #%%
    df['status'] = df['status'].apply(pd.to_numeric)

    #%%
    df
    #%%
    #in this case, I had done the true labels by hand, so let's add them in here
    df_2 = pd.read_csv('new_labels_aug22.csv')
    df['match?'] = df_2['new_label']
    #%%
    df_2
    #%%
    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                                 confusion_matrix, ConfusionMatrixDisplay,
                                 classification_report, accuracy_score)
    import matplotlib.pyplot as plt


    # X = df.drop(columns=['author','VIAF_titlelist','S2_titlelist','overlapping_words','selected_birthyear','overlapping_lemmas','mean_embedding','matched_title?','match','match?','title_list','record_enumerated_titles','S2titles','matched_title_list','normalized_author','common_words','S2_Author','notes','S2_pubdates','S2_Titlelist','selected_birthyear'])  # Drop the label and metadata columns

    X = df.drop(columns=['author','VIAF_titlelist','S2_titlelist','overlapping_words','selected_birthyear','overlapping_lemmas','mean_embedding','match?','S2_pubdates','selected_birthyear','negative_status', 'S2 titlelist','S2_embeddings','VIAF_embeddings'])
    y = df['match?']

    # Optional: If you want to split into training and test sets first
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    #%%
     # df[['birthdate','standard_birthdate','VIAF_birthdate']]
    #%%
    # For DataFrames and Series, you can also check the data types of columns
    print("\nData types in X_train:")
    print(X_train.dtypes)
    print("\nData types in y_train:")
    print(y_train.dtype)
    #%%



    X['birth2mindate'] = pd.to_numeric(X['birth2mindate'], errors= 'coerce')
    X['birth2maxdate'] = pd.to_numeric(X['birth2maxdate'], errors= 'coerce')

    X['abs_birth2mindate'] = pd.to_numeric(X['abs_birth2mindate'], errors= 'coerce')

    X['abs_birth2maxdate'] = pd.to_numeric(X['abs_birth2maxdate'], errors= 'coerce')

    #%%
    pd.to_numeric(X['birth2mindate'], errors= 'coerce')
    #%%
    pd.to_numeric(X['birth2mindate'], errors= 'coerce')
    #%%
    print(X)
    #%%
    # For DataFrames and Series, you can also check the data types of columns
    print("\nData types in X:")
    print(X.dtypes)
    print("\nData types in y:")
    print(y.dtype)
    #%%
    #filter methods of feature selection
    #correlations
    corr_matrix = X.corr()  # Default is Pearson correlation



    #%%
    import seaborn as sns
    plt.figure(figsize=(10, 8))  # Set the figure size
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Correlation Heatmap')
    plt.show()

    #%%
    type(X)
    #%%
    X.dtypes


    #%%
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced_subsample')

    #%%
    y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')

    # Store predictions with original indices
    predictions_df = pd.DataFrame(y_pred_proba, index=original_indices, columns=['Class_0_Proba', 'Class_1_Proba'])

    # Merge the metadata back
    results_df = pd.concat([original_metadata, predictions_df], axis=1)
    results_df['True_Label'] = y.loc[results_df.index]

    # Now you have a DataFrame with metadata and predictions
    print(results_df.head())


    y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')

    # Store predictions with original indices
    predictions_df = pd.DataFrame(y_pred_proba, index=original_indices, columns=['Class_0_Proba', 'Class_1_Proba'])

    # Merge the metadata back
    results_df = pd.concat([original_metadata, predictions_df], axis=1)
    results_df['True_Label'] = y.loc[results_df.index]

    # Now you have a DataFrame with metadata and predictions
    print(results_df.head())
    #%%
    from sklearn.metrics import classification_report

    # Convert predicted probabilities to class labels (0 or 1 based on threshold 0.5)
    # y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)

    y_pred = (y_pred_proba[:, 1] >= 0.8).astype(int)


    # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
    report = classification_report(y, y_pred)

    # Print the classification report
    print(report)
    #%%
    y_pred = (y_pred_proba[:, 1] >= 0.5).astype(int)


    # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
    report = classification_report(y, y_pred)

    # Print the classification report
    print(report)
    #%%
    # y_pred = (y_pred_proba[:, 1] >= 0.7).astype(int)


    # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
    report = classification_report(y, y_pred)

    # Print the classification report
    print(report)

    import pickle
    with open('viaf_classifier_sept23.pkl', 'wb') as file:
        pickle.dump(model, file)
    #%%
print(df['cosine_distance'])

