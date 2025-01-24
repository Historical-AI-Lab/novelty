import pandas as pd





def update_pubdates(name):
    if name in cleaned_dict:
        years = cleaned_dict[name]
        return years

def word_overlap_len(word_overlap):
    return len(word_overlap)




def get_embeddings(titles_list):
    if isinstance(titles_list, str):
        return model.encode(titles_list)
    else:
        return model.encode('')


from ast import literal_eval


def find_max_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return max(pubdates)
    return None


def find_min_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return min(pubdates)
    return None


def birth2maxdate(birth, pubdates):
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
    return None, None


def any_negative(birth2maxdate, birth2mindate):
    if birth2maxdate is not None and birth2mindate is not None:
        if birth2maxdate < 0 or birth2mindate < 0:
            return 1
    return 0


def title_list_len(title_list):
    return len(title_list) if isinstance(title_list, list) else 0


def author_length(author):
    return len(str(author))


# # Master function to process each row
# def process_row(row):
#     birth = row['VIAF_birthdate']
#     pubdates_str = row['S2_pubdates']
#     author = str(row['author'])
#
#     if isinstance(pubdates_str, str) and pubdates_str.strip():
#         try:
#             pubdates = literal_eval(pubdates_str)
#         except (SyntaxError, ValueError):
#             pubdates = None
#         birth2maxdate_value, absbirth2maxdate_value = birth2maxdate(birth, pubdates)
#         birth2mindate_value, absbirth2mindate_value = birth2mindate(birth, pubdates)
#         neg_status = any_negative(birth2maxdate_value, birth2mindate_value)
#         title_length = title_list_len(row['S2Titles'])
#         author_len = author_length(row['author'])
#
#         return {'author': author,
#                 'birth2maxdate': birth2maxdate_value,
#                 'abs_birth2maxdate': absbirth2maxdate_value,
#                 'birth2mindate': birth2mindate_value,
#                 'abs_birth2mindate': absbirth2mindate_value,
#                 'negative_status': neg_status,
#                 'title_length': title_length,
#                 'author_length': author_len,
#                 'S2_pubdates': pubdates
#                 }
#     # else:
#     #     pubdates = None
#
#     elif pubdates_str is None or not isinstance(pubdates_str, list):
#         return {'author':author,
#             'birth2maxdate': None,
#             'abs_birth2maxdate': None,
#             'birth2mindate': None,
#             'abs_birth2mindate': None,
#             'negative_status': None,
#             'title_length': title_list_len(row['S2Titles']),
#             'author_length': author_length(author),
#             'S2_pubdates': None
#
#         }
def avg_pubdate(pubdates):
    if isinstance(pubdates, list) and all(isinstance(pubdate, int) for pubdate in pubdates):
        return sum(pubdates)/len(pubdates)
    else:
        return None

def pub_agef(VIAF_birthdate, avg_pubdate):
    if VIAF_birthdate is not None and avg_pubdate is not None and str(avg_pubdate) != 'nan':
        pub_age = avg_pubdate - VIAF_birthdate
    else:
        pub_age = None
    return pub_age

def define_status(pub_age):
    if pub_age is not None:
        if pub_age < 0:
           status = -1
        elif pub_age > 100:
            status = 1
        elif pub_age < 15:
            status = 2
        else:
            status = 0
            return status
    else:
        status = None
        return status





# Assuming df is your DataFrame

# Print or use result_df as needed
# print(result_df)

if __name__ == '__main__':
    result_df = pd.read_csv('result_df_july18_1136am.csv')
    print(result_df.head())
    result_df['pub_age'] = ""
    print(result_df.head())
    print('pub_age' in result_df.columns)  # Should return True if the column exists

    # file_path = 'LitMetadataWithS2 (3).tsv'
    file_path = '../../metadata/litstudies/LitMetadataWithS2.tsv'
    import pandas as pd

    meta = pd.read_csv(file_path, sep='\t')

    # Use the explode method to expand the authors column
    meta_exploded = meta.explode('authors')

    # Display the exploded DataFrame
    print(meta_exploded)

    meta = meta_exploded

    meta['author'] = meta['authors']

    author_pubyears = {}

    # Iterate over the DataFrame rows
    for index, row in meta.iterrows():
        author = row['author']
        year = row['year']

        # Use setdefault to initialize the list if the author is not already in the dictionary
        author_pubyears.setdefault(author, []).append(year)

    # Display the dictionary to verify
    print(author_pubyears)

    # normalize then apply dictionary
    cleaned_dict = {key.strip("[]'"): value for key, value in author_pubyears.items()}
    result_df['S2_pubdates'] = result_df['author'].apply(update_pubdates)
    #################################################################

    result_df['VIAF_titlelist'] = result_df['record_enumerated_titles']
    result_df['S2_titlelist'] = result_df['S2titles']
    result_df['average_S2_pubdate'] = result_df['avg_pubdates']
    result_df['VIAF_birthdate'] = result_df['standard_birthdate']
    result_df['S2_Author'] = result_df['author']
    # transformations
    result_df['word_overlap'] = result_df['common_words']
    # word_overlap_len -- add?
    result_df['word_overlap_len'] = result_df['word_overlap'].apply(word_overlap_len)
    # cos_sim
    result_df['cos_sim'] = ""
    # birth2maxdate — Maximum Publication Date minus VIAFbirthdate
    # result_df['birth2maxdate'] = result_df[['VIAF_birthdate'],['S2_pubdates'].apply(birth2maxdate)
    # result_df[['birth2maxdate','absbirth2maxdate']] = result_df.apply(
    #     lambda row: birth2maxdate(row['VIAF_birthdate'], row['S2_pubdates']), axis=1
    # )
    result_df[['birth2maxdate', 'absbirth2maxdate']] = result_df.apply(
        lambda row: pd.Series(birth2maxdate(row['VIAF_birthdate'], row['S2_pubdates'])), axis=1
    )
    result_df[['birth2mindate', 'absbirth2mindate']] = result_df.apply(
        lambda row: pd.Series(birth2mindate(row['VIAF_birthdate'], row['S2_pubdates'])), axis=1
    )
    result_df.to_csv('result_df_replication.csv')
    result_df['title_length'] = result_df['S2Titles'].apply(title_list_len)
    result_df['author_length'] = result_df['author'].apply(author_length)
    result_df['avg_pubdate'] = result_df['S2_pubdates'].apply(avg_pubdate)
    # result_df['pub_age'] = result_df.apply(
    #     lambda row: pub_agef(row['birthdate'], row['avg_pubdate']), axis=1)
    print(result_df.head())
    print(type(result_df))
    # result_df['pub_age'] = ""
    print(result_df.head())

    for idx, row in result_df.iterrows():
        if row['VIAF_birthdate'] is not None and row['avg_pubdate'] is not None:
         pub_age = pub_agef(row['VIAF_birthdate'], row['avg_pubdate'])
         result_df.at[idx,'pub_age'] = pub_age
    print(result_df.head())

    # print(pub_age(result_df['VIAF_birthdate'].iloc[0], result_df['avg_pubdate'].iloc[0]))
    result_df['status'] = ""
    result_df['status'] = result_df['pub_age'].apply(define_status)
    result_df.to_csv('result_df_replicated2.csv')

    # # birth2mindate — Minimum Publication Date minus VIAFbirthdate
    # result_df['birth2mindate'] = ""
    # # absbirth2maxdate — abs(Maximum Publication Date minus VIAFbirthdate)
    # result_df['absbirth2maxdate'] = ""
    # # absbirth2mindate — abs(Minimum Publication Date minus VIAFbirthdate)
    # result_df['absbirth2mindate'] = ""
    # # anynegative — if either birth2maxdate or birth2mindate is negative this becomes 1, else 0
    # result_df['any_negative'] = ""
    # # maybe “occupation” or “field of activity” — find out whether they’re common enough
    # result_df['occupation'] = ""
    # # maybe count_of_S2_titles and count_of_VIAF_titles
    # result_df['count_of_S2_titles'] = ""
    # result_df['count_of_VIAF_titles'] = ""
    ################
    # result_df_2 = result_df.apply(process_row, axis=1, result_type='expand')
    # result_df_2.to_csv('result_df_replication.csv')
    #################



