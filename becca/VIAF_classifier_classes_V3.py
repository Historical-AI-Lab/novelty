import pandas as pd
import json
import pickle
import re
import nltk
import spacy
from nltk.corpus import stopwords
from ast import literal_eval

# Initialize Spacy for lemmatization
nlp = spacy.load("en_core_web_sm")

# Download stopwords from NLTK if you haven't already
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

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
        self.meta['author'] = self.meta['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        self.meta_exploded = self.meta.explode('authors')
        for idx, row in self.meta_exploded.iterrows():
            author = normalize_text(str(row['author']))
            self.meta_exploded.at[idx, 'author'] = author

    def process_meta(self):
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

    @staticmethod
    def lemmatize_text(text):
        doc = nlp(text)
        return set(token.lemma_ for token in doc if not token.is_stop and not token.is_punct)

    @staticmethod
    def calculate_lemma_overlap(text1, text2):
        text1 = str(text1)
        text2 = str(text2)
        lemmas1 = AuthorFeatureCreator.lemmatize_text(text1)
        lemmas2 = AuthorFeatureCreator.lemmatize_text(text2)
        overlap = lemmas1.intersection(lemmas2)
        return len(overlap), overlap

    def calculate_birth2maxdate(self):
        self.df['VIAF_birthdate'] = pd.to_numeric(self.df['VIAF_birthdate'], errors='coerce')
        self.df['S2_pubdates'] = pd.to_numeric(self.df['S2_pubdates'], errors='coerce')
        self.df['birth2maxdate'] = self.df['S2_pubdates'] - self.df['VIAF_birthdate']
        self.df['abs_birth2maxdate'] = self.df['birth2maxdate'].abs()

    def calculate_birth2mindate(self):
        self.df['birth2mindate'] = self.df['S2_pubdates'] - self.df['VIAF_birthdate']
        self.df['abs_birth2mindate'] = self.df['birth2mindate'].abs()

    def calculate_word_overlap(self):
        self.df[['word_overlap_count', 'overlapping_words']] = self.df.apply(
            lambda row: self.find_word_overlap(row),
            axis=1
        )

    def calculate_jaccard_distance(self):
        self.df['Jaccard_Distance'] = ""
        for idx, row in self.df.iterrows():
            S2_titlelist = str(row['S2titles']).split(',')
            VIAF_titlelist = str(row['VIAF_titlelist']).split(',')
            jaccard_dist = self.jaccard_distance_for_lists(S2_titlelist, VIAF_titlelist)
            self.df.at[idx, 'Jaccard_Distance'] = jaccard_dist

    def calculate_exact_matches(self):
        self.df['exact_matches'] = ""
        self.df['exact_match_count'] = ""
        for idx, row in self.df.iterrows():
            exact_matches = self.find_exact_matches_for_author(row)
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

    def find_word_overlap(self, row):
        # Implement your word overlap logic here
        # This is a placeholder; you should use your existing find_word_overlap function
        pass

    def jaccard_distance_for_lists(self, list1, list2):
        set1 = set(list1)
        set2 = set(list2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        if union == 0:
            return 0
        return 1 - (intersection / union)

    def find_exact_matches_for_author(self, row):
        matches = []
        author = row['author']
        viaf_titles = row['VIAF_titlelist'] if isinstance(row['VIAF_titlelist'], list) else []
        title_list = row['S2titles'] if isinstance(row['S2titles'], list) else [row['S2titles']]
        viaf_titles_lower = set([str(title).lower() for title in viaf_titles])
        title_list_lower = set([str(title).lower() for title in title_list])
        common_titles = viaf_titles_lower.intersection(title_list_lower)
        if common_titles:
            matches.append({
                'author': author,
                'matching_titles': list(common_titles)
            })
        return matches


# Main execution block
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

    S2_data_dict = {author_name: rows_dict[author_name] for author_name in unique_author_names if author_name in rows_dict}

    processor.update_pubdates(S2_data_dict)

    processor.df['VIAF_titlelist'] = processor.df['viaf_title_list']
    processor.df['S2titles'] = processor.df['title_list']
    processor.df['VIAF_birthdate'] = processor.df['birthdate']

    df = processor.df
    df['avg_pubdate'] = df['S2_pubdates'].apply(find_avg_pubdate)

    feature_creator = AuthorFeatureCreator(df)
    feature_creator.calculate_birth2maxdate()
    feature_creator.calculate_birth2mindate()
    feature_creator.calculate_word_overlap()
    feature_creator.calculate_jaccard_distance()
    feature_creator.calculate_exact_matches()
    # Assuming a model is available
    # feature_creator.generate_embeddings(model)
    # feature_creator.calculate_cosine_distance()
