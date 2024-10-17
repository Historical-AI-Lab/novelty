import pandas as pd



import json
import pickle
import ast
import re
from scipy.spatial.distance import cosine



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

# def clean_pubdates(pubdates):
#     if isinstance(pubdates, list):
#         for pubdate in pubdates:
#             if pubdate == "'no date'":
#                 pubdates.remove(pubdate)
#             pubdate.strip('[]')





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
                if birth2maxdate is not None:
                    abs_birth2maxdate = abs(birth2maxdate)
                    return birth2maxdate, abs_birth2maxdate
                else:
                    return '',''

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
        v_list = ast.literal_eval((row['VIAF_titlelist']))
        s_list = ast.literal_eval(str(row['S2_titlelist']))
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
        v_list = ast.literal_eval(row['VIAF_titlelist'])
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

def cosine_similarity_titles(S2_titlelist, VIAF_titlelist, threshold=0.7):
        # Ensure the title lists are evaluated properly if they are in string format
        if isinstance(S2_titlelist, str):
            S2_titlelist = ast.literal_eval(S2_titlelist)
        if isinstance(VIAF_titlelist, str):
            VIAF_titlelist = ast.literal_eval(VIAF_titlelist)
        VIAF_titles__clean = []
        for title in VIAF_titlelist:
            if not isinstance(title, str):
                title = str(title)
                clean_title = title.replace('"', '').replace("'", '').strip().lower()
                VIAF_titles_clean.append(clean_title)


        # Clean and lower the titles
        S2_titles_clean = [title.replace('"', '').replace("'", '').strip().lower() for title in S2_titlelist]
        VIAF_titles_clean = [title.replace('"', '').replace("'", '').strip().lower() for title in VIAF_titlelist if isinstance(title, str)]

        # Combine both lists to vectorize
        combined_titles = S2_titles_clean + VIAF_titles_clean

        # Use TfidfVectorizer to convert the titles into vectors
        vectorizer = TfidfVectorizer().fit_transform(combined_titles)
        vectors = vectorizer.toarray()

        # Calculate cosine similarity between all pairs
        cosine_sim_matrix = cosine_similarity(vectors)

        # Find matches that meet the similarity threshold
        matches = []
        for i, s2_title in enumerate(S2_titles_clean):
            for j, viaf_title in enumerate(VIAF_titles_clean):
                sim_score = cosine_sim_matrix[i, len(S2_titles_clean) + j]
                if sim_score >= threshold:
                    matches.append((s2_title, viaf_title, sim_score))

        return matches


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
    birth = row['VIAF_birthdate']
    pubdates = row['S2_pubdates']
    # if ',' in pubdates_str:
    #     pubdates = pubdates_str.split(',')
    # else:
    #     if len(pubdates_str)>= 5:
    #         pubdates_str = pubdates_str[:4]
    if pubdates == 'no date' or pubdates == '' or pubdates == 'nan' :
        # if row['S2_Year'] != 'NaN' and row['S2_Year'] != 'nan':
        if pd.notna(row['S2_Year']):
            pubdates = (str(row['S2_Year']))
            pubdates = pubdates[:4]
            pubdates = int(pubdates)
        else:
            pubdates = 0
        if isinstance(birth, str):
            if len(birth) > 4:
                try:
                    birth = int(birth[:4])
                except:
                    birth = int(birth[:3])

            else:
                birth = int(birth)
    author = str(row['author'])
    pubdates, birth = process_pubdates_and_birth(pubdates, birth, author)




    if pubdates is None:



            return {
                    'birth2maxdate': None,
                    'abs_birth2maxdate': None,
                    'birth2mindate': None,
                    'abs_birth2mindate': None,
                    'negative_status': None,
                    'title_count': title_list_len(row['S2_titlelist']),
                    'author_length': author_length(author),
                    'S2_pubdates': row['S2_pubdates'],
                    'VIAF_birthdate': None,
                    'S2_titlelist': row['S2_titlelist'],
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
            #     birth2maxdate_value, absbirth2maxdate_value = birth2maxdate(birth, pubdates, author)
            #     birth2mindate_value, absbirth2mindate_value = birth2mindate(birth, pubdates, author)
                birth2maxdate_value, absbirth2maxdate_value = row['birth2maxdate'], row['abs_birth2maxdate']
                birth2mindate_value, absbirth2mindate_value = row['birth2mindate'], row['abs_birth2mindate']
                neg_status = row['negative_status']
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
                    'S2_pubdates': row['S2_pubdates'],
                    'VIAF_birthdate': birth,
                    'S2 titlelist': row['S2_titlelist'],
                    'VIAF_titlelist': row['VIAF_titlelist'],
                    'author': row['author'],
                    'pub_age' : row['publication_age'],
                   ' avg_pubdate': row['avg_pubdate']

                }
            # else:
            #     print('Test was NaN')

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
        pubdates = ast.literal_eval(pubdates)

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
    title_list = row['S2_titlelist'] if isinstance(row['S2_titlelist'], list) else [row['S2_titlelist']]

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
# def extract_year(df, column_name):
#     # Apply a function to each row of the specified column
#     df[column_name] = df[column_name].apply(lambda x: int(x[:4]) if isinstance(x, str) and len(x) > 8 else x)
#     return df

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)





if __name__ == '__main__':
    # with open('S2_data_dict.txt', 'r') as file:
    #     S2_data_dict = json.load(json_file)
    #load the model for predictions
    # with open('../viaf_classifier_sept23.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)
    # print(loaded_model.feature_names_in_)

    with open('../S2_data_dict.txt', 'r') as filename:
        S2_data_dict = json.load(filename)

    df = pd.read_csv('processed_search_results_VIAF_S2_Oct_all_data.csv')
    # df = df[:100]

    # df = pd.read_csv('../random_sample_search_results_VIAF_S2_Oct.csv', dtype ={'author': 'str', 'record_count': 'Int8', 'record_enumerated': 'Int8', 'viaf_title_list': 'str', 'birthdate': 'str', 'S2_titlelist': 'str', 'S2_pubdates': 'str', 'S2_Year': 'str', 'VIAF_birthdate': 'Int16', 'VIAF_titlelist': 'str'}, usecols = lambda col: col not in ['Search Parameters'])
    # df = pd.read_csv('processed_search_results_VIAF_S2_Oct.csv')
    # df['publication_age'] = ""
    # df['VIAF_birthdate'] = df['birthdate']
    # df['VIAF_titlelist'] = df['viaf_title_list']
    # df = df.drop('avg_pubdates', axis=1)
    # df = df.drop([col for col in df.columns if col.endswith('.1')], axis=1)

    # List of columns to drop
    # columns_to_drop = ['index', 'Unnamed: 0','birthdate','title_list','S2Titles','S2titles','common_words','notes','record_enumerated_titles','selected_birthyear']
    # Drop specified columns
    # df = df.drop(columns=columns_to_drop)

    df['status'] = ""


    for idx, row in df.iterrows():
        if row['publication_age'] < 0:
            df.at[idx, 'status'] = 'not_born'
        elif row['publication_age'] > 100:
            df.at[idx, 'status'] = 'zombie'
        elif row['publication_age'] < 9:
            df.at[idx, 'status'] = 'toddler'
        else:
            df.at[idx, 'status'] = '0'


    df.loc[df['status'] != '0']

    df['status'] = df['status'].str.replace('zombie', '1')
    df['status'] = df['status'].fillna('0')
    df['status'] = df['status'].str.replace('not_born', '2')
    df['status'] = df['status'].str.replace('toddler', '3')

    #
    # Now S2 and VIAF titlelist embeddings
    # %%
    # embeddings

    from sentence_transformers import SentenceTransformer

    # Load the pre-trained model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generate embeddings for each row in the DataFrame
    df['S2_embeddings'] = df['S2_titlelist'].apply(get_embeddings)
    df['VIAF_embeddings'] = df['VIAF_titlelist'].apply(get_embeddings)
    # %%
    # %%

    # %%

    # Function to lemmatize text and return a set of lemmatized words
    import spacy

    nlp = spacy.load("en_core_web_sm")

    df[['lemma_overlap', 'overlapping_lemmas']] = df.apply(
        lambda row: pd.Series(calculate_lemma_overlap(row['VIAF_titlelist'], row['S2_titlelist'])),
        axis=1)


    import pandas as pd
    from nltk.corpus import stopwords
    import nltk

    # Download stopwords from NLTK if you haven't already
    nltk.download('stopwords')
    # Define stop words
    stop_words = set(stopwords.words('english'))

    # Apply the function to each row and create two new columns
    df[['word_overlap_count', 'overlapping_words']] = df.apply(find_word_overlap, axis=1)

    # %%

    df['Jaccard_Distance'] = ""
    df['exact_matches'] = ""
    df['exact_match_count'] = ""

    df.to_csv('df_oct_16_checkpoint.csv')

    # for idx, row in df.iterrows():
    #     if row['word_overlap_count'] > 0:
    #         # print(row['overlapping_words'])
    #         try:
    #             S2_titlelist = ast.literal_eval(row['S2_titlelist'])
    #             VIAF_titlelist = ast.literal_eval(row['VIAF_titlelist'])
    #         except ValueError:
    #             S2_titlelist = str(row['S2_titlelist']).strip('[]').strip('""').split(',')
    #             S2_titles_clean = []
    #             for title in S2_titlelist:
    #                 title = title.replace('"', '').replace("'", '').strip().lower()
    #                 S2_titles_clean.append(title.lower())
    #                 S2_titlelist = S2_titles_clean
    #             VIAF_titlelist = str(row['VIAF_titlelist']).strip('[]').strip('""').split(',')
    #             VIAF_titles_clean = []
    #             for title in VIAF_titlelist:
    #                 title = title.replace('"', '').replace("'", '').strip().lower()
    #                 VIAF_titles_clean.append(title.lower())
    #                 VIAF_titlelist = VIAF_titles_clean
    #         jaccard_distance = jaccard_distance_for_lists(S2_titlelist, VIAF_titlelist)
    #         df.at[idx, 'Jaccard_Distance'] = jaccard_distance
    #         exact_matches = find_exact_matches_for_author(row)
    #         if exact_matches:
    #             print('match')
    #             df.at[idx, 'exact_matches'] = exact_matches

    import ast
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity





    # Example usage on a DataFrame
    for idx, row in df.iterrows():
        if row['word_overlap_count'] > 0:
            try:
                S2_titlelist = ast.literal_eval(row['S2_titlelist'])
                VIAF_titlelist = ast.literal_eval(row['VIAF_titlelist'])
            except ValueError:
                S2_titlelist = str(row['S2_titlelist']).strip('[]').split(',')
                VIAF_titlelist = str(row['VIAF_titlelist']).strip('[]').split(',')
            VIAF_titlelist_clean = []
            # Perform cosine similarity matching between the lists
            cosine_matches = cosine_similarity_titles(S2_titlelist, VIAF_titlelist, threshold=0.7)

            # Example: Add matches to the DataFrame or handle them accordingly
            if cosine_matches:
                # print(f"Matches found for row {idx}: {cosine_matches}")
                df.at[idx, 'cosine_matches'] = len(cosine_matches)
            else:
                df.at[idx, 'cosine_matches'] = 0

                # print(f"No matches found for row {idx}")

    df['cosine_distance'] = ""

    for idx, row in df.iterrows():
        VIAF_embedding = row['VIAF_embeddings']
        S2_embedding = row['S2_embeddings']
        cosine_dist = get_cosine_distance_bw_title_embeddings(S2_embedding, VIAF_embedding)
        df.at[idx, 'cosine_distance'] = cosine_dist


    # %%
    # rename to keep track of where birthdate data came from
    # df['VIAF_birthdate'] = df['birthyear']
    # for idx, row in df.iterrows():
    #     if row['VIAF_birthdate'] == '1949-06-01':
    #         row['VIAF_birthdate'] = '1949'

    df.loc[df['VIAF_birthdate'] == '1949-06-01', 'VIAF_birthdate'] = 1949
    df.loc[df['S2_pubdates'] == '1949-06-01', 'S2_pubdates'] = 1949
    # df.loc[df['birthyear'] == '1949-06-01', 'birthyear'] = 1949

    #
    # df_notnull_birth2max = df.loc[df['birth2maxdate'].notnull()]
    # print(df_notnull_birth2max.head(30))
    df.to_csv('all_data_get_features_asCSV.csv')