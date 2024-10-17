
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





import pandas as pd
import ast

df = pd.read_csv('df_oct_16_checkpoint.csv')

df.head(30)

# df = apply_jaccard(df, 'S2_titlelist', 'VIAF_titlelist')

for idx, row in df.iterrows():
    if row['word_overlap_count'] > 0:
        # print(row['overlapping_words'])
        try:
            S2_titlelist = ast.literal_eval(row['S2_titlelist'])
            VIAF_titlelist = ast.literal_eval(row['VIAF_titlelist'])
        except ValueError:
            S2_titlelist = str(row['S2_titlelist']).strip('[]').strip('""').split(',')
            S2_titles_clean = []
            for title in S2_titlelist:
                title = title.replace('"', '').replace("'", '').strip().lower()
                S2_titles_clean.append(title.lower())
                S2_titlelist = S2_titles_clean
            VIAF_titlelist = str(row['VIAF_titlelist']).strip('[]').strip('""').split(',')
            VIAF_titles_clean = []
            for title in VIAF_titlelist:
                title = title.replace('"', '').replace("'", '').strip().lower()
                VIAF_titles_clean.append(title.lower())
                VIAF_titlelist = VIAF_titles_clean
        jaccard_distance = jaccard_distance_for_lists(S2_titlelist, VIAF_titlelist)
        df.at[idx, 'Jaccard_Distance'] = jaccard_distance
        exact_matches = find_exact_matches_for_author(row)
        if exact_matches:
            print('match')
        # df.at[idx, 'exact_matches'] = exact_matches
        # df.at[idx, 'exact_match_count'] = len(exact_matches)

        import ast
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity


        def cosine_similarity_titles(S2_titlelist, VIAF_titlelist, threshold=0.7):
            # Ensure the title lists are evaluated properly if they are in string format
            if isinstance(S2_titlelist, str):
                S2_titlelist = ast.literal_eval(S2_titlelist)
            if isinstance(VIAF_titlelist, str):
                VIAF_titlelist = ast.literal_eval(VIAF_titlelist)

            # Clean and lower the titles
            S2_titles_clean = [title.replace('"', '').replace("'", '').strip().lower() for title in S2_titlelist]
            VIAF_titles_clean = [title.replace('"', '').replace("'", '').strip().lower() for title in VIAF_titlelist]

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


        # Example usage on a DataFrame
        for idx, row in df.iterrows():
            if row['word_overlap_count'] > 0:
                try:
                    S2_titlelist = ast.literal_eval(row['S2_titlelist'])
                    VIAF_titlelist = ast.literal_eval(row['VIAF_titlelist'])
                except ValueError:
                    S2_titlelist = str(row['S2_titlelist']).strip('[]').split(',')
                    VIAF_titlelist = str(row['VIAF_titlelist']).strip('[]').split(',')

                # Perform cosine similarity matching between the lists
                cosine_matches = cosine_similarity_titles(S2_titlelist, VIAF_titlelist, threshold=0.7)

                # Example: Add matches to the DataFrame or handle them accordingly
                if cosine_matches:
                    # print(f"Matches found for row {idx}: {cosine_matches}")
                    df.at[idx, 'cosine_matches'] = len(cosine_matches)
                else:
                    df.at[idx, 'cosine_matches'] = 0

                    # print(f"No matches found for row {idx}")


