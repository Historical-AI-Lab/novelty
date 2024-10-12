import json
import pandas as pd
import pickle

import pandas as pd
from nltk.corpus import stopwords
import nltk

# Download stopwords from NLTK if you haven't already
nltk.download('stopwords')
# Define stop words
stop_words = set(stopwords.words('english'))


### UTILITY FUNCTIONS

def load_json(filepath):
    """Load a JSON file."""
    with open(filepath, 'r') as file:
        return json.load(file)

def save_json(data, filepath):
    """Save data to a JSON file."""
    with open(filepath, 'w') as file:
        json.dump(data, file, indent=4)

def normalize_text(text):
    """Normalize text for comparison (you can define this based on your needs)."""
    return str(text).strip().lower()

def is_author_in_dict(author, author_dict):
    """Check if an author exists in a given dictionary."""
    return author in author_dict

def find_avg_pubdate(pubdates):
    """Find the average publication date."""
    return sum(pubdates) / len(pubdates) if pubdates else None

#DATAFRAME OPERATIONS 1,1

def process_author_rows(meta):
    """Process author rows from the metadata."""
    rows = []
    for index, row in meta.iterrows():
        author = normalize_text(row['author'])
        author_list = author.split(',') if ',' in author else [author]
        for author in author_list:
            rows.append({
                'author': author,
                'journal': row['journal'],
                'year': row['year'],
                'title': row['title'],
                'S2titles': row['S2titles'],
                'S2Years': row['S2years'],
            })
    return rows

def update_df_with_s2_data(df, s2_data_dict):
    """Update DataFrame with S2 publication dates and titles."""
    for idx, row in df.iterrows():
        author = row['author']
        if is_author_in_dict(author, s2_data_dict):
            pubdate = s2_data_dict[author]['S2Years']
            df.at[idx, 'S2_pubdates'] = pubdate
            pubdate2 = s2_data_dict[author]['year']
            df.at[idx, 'S2_Year'] = pubdate2
    return df

#DATAFRAME PROCESSING
class AuthorDataProcessor:
    def __init__(self, df, meta_filepath, model_filepath):
        self.df = df
        self.meta = pd.read_csv(meta_filepath, sep='\t')
        self.model = self.load_model(model_filepath)
        self.s2_data_dict = {}

    def load_model(self, filepath):
        """Load a machine learning model from file."""
        with open(filepath, 'rb') as file:
            return pickle.load(file)

    def process_meta(self):
        """Process metadata for S2 data."""
        self.meta['author'] = self.meta['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        meta_exploded = self.meta.explode('authors')
        self.meta = meta_exploded
        self.meta['author'] = self.meta['author'].apply(normalize_text)
        rows = process_author_rows(self.meta)
        self.s2_data_dict = {row['author']: row for row in rows}

    def update_df(self):
        """Update the main DataFrame with S2 and VIAF data."""
        self.df['author'] = self.df['author'].apply(normalize_text)
        self.df = update_df_with_s2_data(self.df, self.s2_data_dict)

    def predict(self):
        """Run the model on the DataFrame and add predictions."""
        self.df['predictions'] = self.model.predict(self.df)

    def save_results(self, output_filepath):
        """Save the processed DataFrame to a CSV file."""
        self.df.to_csv(output_filepath, index=False)

    import pandas as pd
    from nltk.corpus import stopwords
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    import spacy
    import nltk

    # Download stopwords and load Spacy model
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    nlp = spacy.load("en_core_web_sm")

class AuthorFeatureCreator:
        def __init__(self, df):
            """
            Initialize the processor with the DataFrame containing author data.
            :param df: Input DataFrame with author and title data.
            """
            self.df = df

        @staticmethod
        def find_word_overlap(s2_titlelist, viaf_titlelist):
            """
            Calculate word overlap between two title lists, excluding stopwords.
            """
            s2_set = set(str(s2_titlelist).lower().split(',')).difference(stop_words)
            viaf_set = set(str(viaf_titlelist).lower().split(',')).difference(stop_words)
            overlap = s2_set.intersection(viaf_set)
            return len(overlap), overlap

        @staticmethod
        def jaccard_distance_for_lists(list1, list2):
            """
            Compute the Jaccard distance between two lists.
            """
            set1 = set(list1)
            set2 = set(list2)
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return 1 - (intersection / union) if union != 0 else 1

        @staticmethod
        def get_cosine_distance_bw_embeddings(embedding1, embedding2):
            """
            Compute cosine distance between two embeddings.
            """
            embedding1 = np.array(embedding1).reshape(1, -1)
            embedding2 = np.array(embedding2).reshape(1, -1)
            cosine_sim = cosine_similarity(embedding1, embedding2)[0][0]
            return 1 - cosine_sim

        @staticmethod
        def find_exact_matches(s2_titlelist, viaf_titlelist):
            """
            Find exact title matches between two title lists.
            """
            s2_set = set(str(s2_titlelist).split(','))
            viaf_set = set(str(viaf_titlelist).split(','))
            exact_matches = s2_set.intersection(viaf_set)
            return list(exact_matches)

        @staticmethod
        def calculate_lemma_overlap(viaf_titlelist, s2_titlelist):
            """
            Calculate overlap of lemmatized words between two title lists.
            """
            viaf_lemmas = set([token.lemma_ for token in nlp(str(viaf_titlelist)) if not token.is_stop])
            s2_lemmas = set([token.lemma_ for token in nlp(str(s2_titlelist)) if not token.is_stop])
            overlap = viaf_lemmas.intersection(s2_lemmas)
            return len(overlap), overlap

        def process_data(self):
            """
            Process the data to compute all features: word overlap, Jaccard distance,
            cosine distance, exact matches, and lemma overlap.
            """
            self.df['word_overlap_count'], self.df['overlapping_words'] = zip(*self.df.apply(
                lambda row: self.find_word_overlap(row['S2Titles'], row['VIAF_Titlelist']), axis=1))

            self.df['Jaccard_Distance'] = self.df.apply(
                lambda row: self.jaccard_distance_for_lists(str(row['S2_Titlelist']).split(','),
                                                            str(row['VIAF_Titlelist']).split(',')), axis=1)

            self.df['cosine_distance'] = self.df.apply(
                lambda row: self.get_cosine_distance_bw_embeddings(row['S2_embeddings'], row['VIAF_embeddings']),
                axis=1)

            self.df['exact_matches'] = self.df.apply(
                lambda row: self.find_exact_matches(row['S2_Titlelist'], row['VIAF_Titlelist']), axis=1)

            self.df['lemma_overlap'], self.df['overlapping_lemmas'] = zip(*self.df.apply(
                lambda row: self.calculate_lemma_overlap(row['VIAF_Titlelist'], row['S2_Titlelist']), axis=1))

        def get_dataframe(self):
            """
            Return the processed DataFrame.
            """
            return self.df


if __name__ == '__main__':

    # Initialize and run the processor
    df = pd.read_csv('all_search_results_df_18hr_sept17.csv').iloc[:100]
    df['S2Titles'] = df['title_list']
    processor = AuthorDataProcessor(df, 'Oct_Classifier/LitMetadataWithS2 (3).tsv', 'viaf_classifier_sept23.pkl')

    # Process metadata and update the dataframe
    processor.process_meta()
    processor.update_df()

    # Assuming df is your DataFrame with the necessary columns
    feature_creation = AuthorFeatureCreator(df)

    # Process the data to calculate all the features
    feature_creation.process_data()

    # Retrieve the processed DataFrame with new features
    processed_df = feature_creation.get_dataframe()

    # Print the processed DataFrame
    print(processed_df.head())

    # Run the predictions
    # processor.predict()
    # Define the feature columns (you may need to adjust based on your model's input)
    feature_columns = ['S2_Titlelist', 'VIAF_Titlelist', 'publication_age', 'overlapping_words', 'cosine_distance']

    # x is the DataFrame with only the feature columns
    x = df[feature_columns]

    # Define the target column
    y = df['target_column_name']

    # Train the model (for example, during model training)
    model.fit(x, y)

    # Predicting using the loaded model
    predictions = loaded_model.predict(x)

    # Adding predictions to the DataFrame
    df['predictions'] = predictions

    # Save the updated DataFrame
    processor.save_results('processed_results.csv')


