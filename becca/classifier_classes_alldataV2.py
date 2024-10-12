import pandas as pd
import json
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
import nltk
from ast import literal_eval

import spacy

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


def find_avg_pubdate(pubdates):
    try:
        if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
            avg_pubdate =  (sum(pubdates) / len(pubdates))
        elif isinstance(pubdates, int):
            avg_pubdate = pubdates
        return avg_pubdate
    except:
        return None

# Download stopwords from NLTK if you haven't already
nltk.download('stopwords')
# Define stop words
stop_words = set(stopwords.words('english'))
def find_exact_matches_for_author(row):
    matches = []

    # Iterate over each row in the DataFrame
    # for idx, row in df.iterrows():
    author = row['author']

    # Get titles from viaf_title_list and title_list
    viaf_titles = row['VIAF_titlelist'] if isinstance(row['VIAF_titlelist'], list) else []
    title_list = row['S2titles'] if isinstance(row['S2titles'], list) else [row['S2titles']]

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

def count_matches(exact_title_match):
    return len(exact_title_match)

def extract_year(df, column_name):
    # Apply a function to each row of the specified column
    df[column_name] = df[column_name].apply(lambda x: int(x[:4]) if isinstance(x, str) and len(x) > 8 else x)
    return df


# Function to create bag of words after removing stop words
def bag_of_words(title):
    words = title.lower().split()
    return set([word for word in words if word not in stop_words])

# Function to find and count word overlap
def find_word_overlap(row):
    if not pd.isna(row['VIAF_titlelist']) and not pd.isna(row['S2titles']):
        # try:
        # Convert string representations to lists using ast.literal_eval
        v_list = literal_eval(row['VIAF_titlelist'])
        s_list = str(row['S2titles'])
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

class AuthorDataProcessor:
    def __init__(self, search_results_path, meta_path, viaf_file1, viaf_file2):
        self.df = pd.read_csv(search_results_path)
        self.meta = pd.read_csv(meta_path, sep='\t')
        self.meta_exploded = None
        self.new_viaf_dict = self.load_viaf_data(viaf_file1)
        self.new_viaf_dict2 = self.load_viaf_data(viaf_file2)

    def load_viaf_data(self, file_path):
        with open(file_path, 'r') as file:
            return json.load(file)



    def normalize_authors(self):
        self.meta['author'] = self.meta['authors'].apply(
            lambda x: eval(x) if isinstance(x, str) else x)
        self.meta_exploded = self.meta.explode('authors')
        for idx, row in self.meta_exploded.iterrows():
            author = normalize_text(str(row['author']))
            self.meta_exploded.at[idx, 'author'] = author

    def process_meta(self):
        # Converts the exploded meta into normalized authors and rows
        rows = []
        for index, row in self.meta_exploded.iterrows():
            rows += self.create_rows(row)
        return rows

    def create_rows(self, row):
        author = normalize_text(row['author'])
        if ',' in author:
            return [{
                'author': a,
                'journal': row['journal'],
                'year': row['year'],
                'title': row['title'],
                'S2titles': row['S2titles'],
                'S2Years': row['S2years'],
            } for a in author.split(',')]
        else:
            return [{
                'author': author,
                'journal': row['journal'],
                'year': row['year'],
                'title': row['title'],
                'S2titles': row['S2titles'],
                'S2Years': row['S2years'],
            }]

    def update_pubdates(self, S2_data_dict):
        self.df['S2_pubdates'] = ""
        for idx, row in self.df.iterrows():
            author = row['author']
            if author in S2_data_dict:
                self.df.at[idx, 'S2_pubdates'] = S2_data_dict[author]['S2Years']

    def add_model_predictions(self, model_path):
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        predictions = model.predict(self.df)
        self.df['predictions'] = predictions



class AuthorFeatureCreator:
    def __init__(self, df):
        self.df = df

    def calculate_birth2maxdate(self):
        # Ensure that birthyear and pubdates are numeric
        self.df['VIAF_birthdate'] = pd.to_numeric(self.df['VIAF_birthdate'], errors='coerce')
        self.df['S2_pubdates'] = pd.to_numeric(self.df['S2_pubdates'], errors='coerce')

        # Calculate max publication date difference
        self.df['birth2maxdate'] = self.df['S2_pubdates'] - self.df['VIAF_birthdate']
        self.df['abs_birth2maxdate'] = self.df['birth2maxdate'].abs()

    # Method to calculate the difference between birthdate and min publication date
    def calculate_birth2mindate(self):
        # Assuming 'S2_pubdates' already contains both min and max pubdates (or filter accordingly)
        self.df['birth2mindate'] = self.df['S2_pubdates'] - self.df['VIAF_birthdate']
        self.df['abs_birth2mindate'] = self.df['birth2mindate'].abs()

    def calculate_word_overlap(self, stop_words):
        nltk.download('stopwords')
        stop_words = set(stopwords.words('english'))
        self.df[['word_overlap_count', 'overlapping_words']] = self.df.apply(
            lambda row: find_word_overlap(row),
            axis=1
        )

    def calculate_jaccard_distance(self):
        self.df['Jaccard_Distance'] = ""
        for idx, row in self.df.iterrows():
            S2_titlelist = str(row['S2titles']).split(',')
            VIAF_titlelist = str(row['VIAF_titlelist']).split(',')
            jaccard_dist = jaccard_distance_for_lists(S2_titlelist, VIAF_titlelist)
            self.df.at[idx, 'Jaccard_Distance'] = jaccard_dist

    def calculate_exact_matches(self):
        self.df['exact_matches'] = ""
        self.df['exact_match_count'] = ""
        for idx, row in self.df.iterrows():
            exact_matches = find_exact_matches_for_author(row)
            self.df.at[idx, 'exact_matches'] = exact_matches
            self.df.at[idx, 'exact_match_count'] = len(exact_matches)

    def generate_embeddings(self, model):
        self.df['S2_embeddings'] = self.df['S2titles'].apply(get_embeddings, model=model)
        self.df['VIAF_embeddings'] = self.df['VIAF_titlelist'].apply(get_embeddings, model=model)

    def calculate_cosine_distance(self):
        self.df['cosine_distance'] = ""
        for idx, row in self.df.iterrows():
            cosine_dist = get_cosine_distance_bw_title_embeddings(row['S2_embeddings'], row['VIAF_embeddings'])
            self.df.at[idx, 'cosine_distance'] = cosine_dist


if __name__ == '__main__':
    df = pd.read_csv('all_search_results_df_18hr_sept17.csv')
    df = df.iloc[:10]


    processor = AuthorDataProcessor(
        search_results_path='search_results_2.csv',
        meta_path='Oct_Classifier/LitMetadataWithS2 (3).tsv',
        viaf_file1='new_viaf_data.txt',
        viaf_file2='new_viaf_data_2.txt'


    )


    processor.normalize_authors()
    rows = processor.process_meta()

    rows_dict = {row['author']: row for row in rows}

    df_search_results = pd.read_csv('search_results_2.csv')
    unique_author_names = []

    for author_name in df_search_results['author']:
        if author_name not in unique_author_names:
            unique_author_names.append(author_name)

    # Step 2: Iterate through the unique author names and retrieve the corresponding row from the dictionary
    S2_data_dict = {author_name: rows_dict[author_name] for author_name in unique_author_names if
                    author_name in rows_dict}

    # Update publication dates
    processor.update_pubdates(S2_data_dict)

    processor.df['VIAF_titlelist'] = processor.df['viaf_title_list']
    processor.df['S2titles'] = processor.df['title_list']
    processor.df['VIAF_birthdate'] = processor.df['birthdate']

    df = processor.df

    df['avg_pubdate'] = df['S2_pubdates'].apply(find_avg_pubdate)


    df['publication_age'] = ""
    # df['publication_age'] = df['avg_pubdate'] - df['birthyear']
    for idx, row in df.iterrows():
        avg_pubdate = row['avg_pubdate']
        birth = str(row['VIAF_birthdate'])
        if len(birth) >= 8:
            birth = birth[:4]
            if avg_pubdate is None:
                break
            else:
                pub_age = avg_pubdate - int(birth)
                df.at[idx, 'publication_age'] = pub_age

    df['pub_age'] = df['publication_age']
    # %%

    df['status'] = ""

    df['publication_age'] = pd.to_numeric(df['publication_age'], errors='coerce')


    for idx, row in df.iterrows():
        if row['publication_age'] < 0:
            df.at[idx, 'status'] = 'not_born'
        elif row['publication_age'] > 100:
            df.at[idx, 'status'] = 'zombie'
        elif row['publication_age'] < 9:
            df.at[idx, 'status'] = 'toddler'
        else:
            df.at[idx, 'status'] = '0'

    # %%
    # df.loc[df['status'] != '0']
    # %%
    df['status'] = df['status'].str.replace('zombie', '1')
    df['status'] = df['status'].fillna('0')
    df['status'] = df['status'].str.replace('not_born', '2')
    df['status'] = df['status'].str.replace('toddler', '3')

    df['avg_pubdate'] = df['S2_pubdates'].apply(find_avg_pubdate)

    df[['lemma_overlap', 'overlapping_lemmas']] = df.apply(
        lambda row: pd.Series(calculate_lemma_overlap(row['VIAF_titlelist'], row['S2titles'])),
        axis=1)

    feature_creator = AuthorFeatureCreator(processor.df)
    feature_creator.calculate_birth2maxdate()
    feature_creator.calculate_birth2mindate()
    feature_creator.calculate_word_overlap(stop_words=set(stopwords.words('english')))
    feature_creator.calculate_jaccard_distance()
    feature_creator.calculate_exact_matches()

    original_indices = processor.df.index
    original_metadata = processor.df[
        ['VIAF_titlelist', 'author', 'S2titles', 'status', 'pub_age', 'avg_pubdate',
         'VIAF_birthdate', 'overlapping_words', 'word_overlap_count', 'lemma_overlap', 'overlapping_lemmas',
         'exact_matches']].copy()
    print(processor.df.columns.tolist())
    columns_to_drop = ['S2 titlelist', 'S2_embeddings', 'S2_pubdates', 'VIAF_embeddings', 'S2_titlelist',
                       'VIAF_titlelist', 'author', 'mean_embedding', 'negative_status', 'overlapping_lemmas',
                       'overlapping_words', 'exact_matches']

    # Check which columns actually exist in the DataFrame before dropping
    existing_columns_to_drop = [col for col in columns_to_drop if col in processor.df.columns]
    # Drop existing columns
    processor.df.drop(columns=existing_columns_to_drop, inplace=True)
    print("\nData types in df:")
    print(df.dtypes)

    df = processor.df
    df['birth2mindate'] = pd.to_numeric(df['birth2mindate'], errors='coerce')
    df['birth2maxdate'] = pd.to_numeric(df['birth2maxdate'], errors='coerce')
    df['abs_birth2mindate'] = pd.to_numeric(df['abs_birth2mindate'], errors='coerce')
    df['abs_birth2maxdate'] = pd.to_numeric(df['abs_birth2maxdate'], errors='coerce')

    # Add model predictions
    processor.add_model_predictions('viaf_classifier_sept23.pkl')
