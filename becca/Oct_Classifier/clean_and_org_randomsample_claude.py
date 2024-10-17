import pandas as pd
import numpy as np
from ast import literal_eval
from datetime import datetime
import re
import nltk
import spacy
from nltk.corpus import stopwords
from scipy.spatial.distance import cosine

# from becca.clean_viaf_classifier2 import clean_pubdates

# Download required NLTK data
nltk.download('stopwords')

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define constants
STOP_WORDS = set(stopwords.words('english'))


def normalize_text(text):
    if pd.isna(text):
        return ''
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s,]', '', text)
    return text.strip()


def process_pubdates(pubdates_str):
    if isinstance(pubdates_str, str):
        if ',' in pubdates_str:
            return [int(year.strip()[:4]) for year in pubdates_str.split(',') if year.strip()[:4].isdigit()]
        elif pubdates_str.strip()[:4].isdigit():
            return [int(pubdates_str[:4])]
    return []


def process_birth(birth):
    if isinstance(birth, str):
        birth = birth.replace('-', '')[:4]
    # return int(birth) if birth.isdigit() else None
    return int(birth)


def calculate_age_metrics(birth, pubdates):
    if birth and pubdates:
        max_pubdate = max(pubdates)
        min_pubdate = min(pubdates)
        birth2max = max_pubdate - birth
        birth2min = min_pubdate - birth
        return birth2max, abs(birth2max), birth2min, abs(birth2min), any(x < 0 for x in [birth2max, birth2min])
    return None, None, None, None, None


def process_row(row):
    birth = process_birth(row['birthdate'])
    pubdates = process_pubdates(row['S2_Year'])

    birth2max, abs_birth2max, birth2min, abs_birth2min, negative_status = calculate_age_metrics(birth, pubdates)

    return {
        'birth2maxdate': birth2max,
        'abs_birth2maxdate': abs_birth2max,
        'birth2mindate': birth2min,
        'abs_birth2mindate': abs_birth2min,
        'negative_status': negative_status,
        'title_count': len(row['S2_titlelist']) if isinstance(row['S2_titlelist'], list) else 0,
        'author_length': len(row['author']),
        'S2_pubdates': pubdates,
        'VIAF_birthdate': birth,
        'S2_titlelist': row['S2_titlelist'],
        'VIAF_titlelist': row['VIAF_titlelist'],
        'author': row['author'],
        'pub_age': row['publication_age'],
        'avg_pubdate': np.mean(pubdates) if pubdates else None
    }

def find_avg_pubdate(pubdates):
        if isinstance(pubdates, int):
            avg_pubdate = pubdates
            return avg_pubdate, pubdates
        if isinstance(pubdates, float):
            pubdates = str(pubdates)
            if len(pubdates) > 5:
                pubdates = pubdates[:4]
                # pubdates = int(pubdates)
                if 'no d' not in pubdates:
                    avg_pubdate = int(pubdates)
                    return avg_pubdate, pubdates
                else:
                    avg_pubdate = 0
                    return avg_pubdate,pubdates
        if isinstance(pubdates, str) and pubdates != 'no date' and pubdates != "" and pubdates != 'nan' and pubdates != '[]':
            # pubdates = pubdates.strip('-')
            pubdates = pubdates.replace('-', '')
            if '['in pubdates:
                try:
                    clean_pubdates = []
                    pubdates = pubdates.strip('[]')
                    pubdates = pubdates.split(',')
                    for pubdate in pubdates:
                        # pubdate = pubdate.strip(' ')
                        pubdate = pubdate.strip("''")
                        if pubdate == 'no d' or pubdate == 'no date' or pubdate == "'no date'":
                            pubdates.remove(pubdate)
                        else:
                            pubdate = pubdate.strip("''")
                            if len(pubdate) > 5:
                                pubdate = pubdate[:4]
                                clean_pubdates.append(int(pubdate))
                                pubdates = clean_pubdates
                                avg_pubdate = (sum(pubdates) / len(pubdates))
                                return avg_pubdate, pubdates

                except:
                    # if len(pubdates) > 5:
                    #     pubdates = pubdates[:4]
                        try:
                            if "'no date'" not in pubdates and 'no date' not in pubdates:
                                if isinstance(pubdates, list):
                                    for pubdate in pubdates:
                                        # pubdate = pubdate.strip(' ')
                                        pubdate = pubdate.strip("''")
                                        if pubdate == 'no d' or pubdate == 'no date' or pubdate == "'no date'":
                                            pubdates.remove(pubdate)
                                        else:
                                            pubdate = pubdate.strip("''")
                                            pubdate = pubdate.strip(" ")
                                            pubdate = pubdate.strip("'")
                                            if len(pubdate) > 5:
                                                pubdate = pubdate[:4]
                                                clean_pubdates.append(int(pubdate))
                                                pubdates = clean_pubdates
                                                avg_pubdate = (sum(pubdates) / len(pubdates))
                                                return avg_pubdate
                                else:
                                    if 'no d' not in pubdates:
                                        pubdates = int(pubdates)
                                        avg_pubdate = pubdates
                                        return avg_pubdate
                                    else:
                                        avg_pubdate = 0
                                        return avg_pubdate
                            else:
                                pubdates.remove("'no date'")
                                if isinstance(pubdates, list):
                                    for pubdate in pubdates:
                                        # pubdate = pubdate.strip(' ')
                                        pubdate = pubdate.strip("''")
                                        if pubdate == 'no d' or pubdate == 'no date' or pubdate == "'no date'":
                                            pubdates.remove(pubdate)
                                        else:
                                            pubdate = pubdate.strip("''")
                                            pubdate = pubdate.strip(" ")
                                            pubdate = pubdate.strip("'")
                                            if len(pubdate) > 5:
                                                pubdate = pubdate[:4]
                                                clean_pubdates.append(int(pubdate))
                                                pubdates = clean_pubdates
                                                avg_pubdate = (sum(pubdates) / len(pubdates))
                                                return avg_pubdate
                                # pubdates = int(pubdates)
                                # avg_pubdate = pubdates
                                # return avg_pubdate
                        except ValueError as e:
                            # print(f"Error converting pubdates to int: {e}")
                            avg_pubdate = 0
                            return avg_pubdate
                        # avg_pubdate = pubdates
            # return avg_pubdate
        if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates) and len(pubdates) > 1:
            avg_pubdate = (sum(pubdates) / len(pubdates))
        else:
            avg_pubdate = 0
        return avg_pubdate


def main():
    # Load data
    meta = pd.read_csv('LitMetadataWithS2 (3).tsv', sep='\t')
    meta['author'] = meta['authors'].apply(lambda x: eval(x) if isinstance(x, str) else x)
    meta = meta.explode('authors')
    meta['author'] = meta['author'].astype(str).apply(normalize_text)

    df = pd.read_csv('../result_df_july18_1136am.csv')
    df['author'] = df['author'].apply(normalize_text)

    # Process author data
    author_dict = {}
    for _, row in meta.iterrows():
        author = row['author']
        if author not in author_dict:
            author_dict[author] = []
        author_dict[author].append({
            'journal': row['journal'],
            'year': row['year'],
            'title': row['title'],
            'S2titles': row['S2titles'],
            'S2years': row['S2years']
        })

    # Update df with S2 data
    df['S2_titlelist'] = df['author'].map(lambda x: [entry['S2titles'] for entry in author_dict.get(x, [])])
    df['S2_pubdates'] = df['author'].map(lambda x: [entry['S2years'] for entry in author_dict.get(x, [])])
    df['S2_Year'] = df['author'].map(lambda x: [entry['year'] for entry in author_dict.get(x, [])])

    df['VIAF_titlelist'] = df['record_enumerated_titles']
    # df['avg_pubdate'] = df['S2_pubdates'].apply(find_avg_pubdate)
    for idx, row in df.iterrows():
        pubdates = str(row['S2_pubdates'])
        birth = str(row['birthdate'])
        if pubdates is not None or pubdates != 'nan' or pubdates != '[]':
            try:
                avg_pubdate, pubdates = find_avg_pubdate(pubdates)

            except:
                avg_pubdate = 0
                pubdates = []
                # print('error?')
        else:
            avg_pubdate = 0
        df.at[idx, 'avg_pubdate'] = avg_pubdate
        if pubdates != '[]':
            birth2max, abs_birth2max, birth2min, abs_birth2min, negative_status = calculate_age_metrics(int(birth), pubdates)
            df.at[idx, 'birth2maxdate'] = birth2max
            df.at[idx, 'birth2mindate'] = birth2min
            df.at[idx, 'abs_birth2maxdate'] = abs_birth2max
            df.at[idx, 'abs_birth2mindate'] = abs_birth2min


    # %%
    # df_notnull = df.loc[df['avg_pubdate'].notnull()]
    # # %%
    # df_notnull

    # %%
    # publication_age
    df['publication_age'] = ""
    # df['publication_age'] = df['avg_pubdate'] - df['birthyear']
    for idx, row in df.iterrows():
        avg_pubdate = row['avg_pubdate']
        birth = str(row['birthdate'])
        if len(birth) >= 5:
            birth = birth[:4]
            birth = birth.strip('-')
        else:
            continue
        if avg_pubdate is None or avg_pubdate == 'error':
            break
        else:
            pub_age = int(avg_pubdate) - int(birth)
            df.at[idx, 'publication_age'] = pub_age
    # Process each row
    processed_data = df.apply(process_row, axis=1)
    df_processed = pd.DataFrame(processed_data.tolist())

    # Combine processed data with original dataframe
    df_final = pd.concat([df, df_processed], axis=1)


    #make sure the S2 pubdates didn't get messed up...
    # df_final['S2_pubdates'] = df_final['author'].map(lambda x: [entry['S2years'] for entry in author_dict.get(x, [])])


    # Save results
    df_final.to_csv('processed_search_results_VIAF_S2_Oct.csv', index=False)


if __name__ == '__main__':
    main()
    print("Processing complete. Results saved to 'processed_search_results_VIAF_S2_Oct.csv'")