import pandas as pd
import pickle
import numpy as np

# def clean_up_birthdate(birthdate):
#     if len(birthdate) > 5:
#         birthdate = birthdate[:4]
#     else:
#         birthdate = birthdate
#     return birthdate


def main(df_5, df_6, loaded_model):
#
    def clean_birthdate2(date):
        if pd.isna(date) or date == 0:
            return np.nan
        try:
            # Remove any non-numeric characters except '-'
            date = ''.join(char for char in str(date) if char.isdigit() or char == '-')
            # If it ends with '-', remove it
            if date.endswith('-'):
                date = date[:-1]
            # Convert to integer
            return int(date)
        except ValueError:
            return np.nan


    # df = pd.read_csv('../../all_data_get_features_asCSV.csv')
    df = df_6

    for idx, row in df.iterrows():
        sim_scores = row['sim_scores']
        num_list = []
        if str(sim_scores).lower() != 'nan':
            for x in sim_scores:
                x = x.strip()  # Remove any extra spaces
                try:
                    # Convert to float and add to list (skip 'NaN')
                    if x.lower() != 'nan':
                        num_list.append(float(x))
                except ValueError:
                    continue  # Skip any bad data

            # Calculate the mean of the numbers, handle empty lists
            if num_list:
                mean_score = np.mean(num_list)  # Compute mean if list is not empty
            else:
                mean_score = 0  # Default to 0 if no valid numbers

            # Store the mean in a new column
            df.at[idx, 'sim_scores_mean'] = mean_score


    df['sim_scores_mean'] = df['sim_scores_mean'].fillna(0)


    # Apply the cleaning function
    # df['VIAF_birthdate'] = df['VIAF_birthdate'].apply(clean_up_birthdate)
    df['VIAF_birthdate'] = df['VIAF_birthdate'].apply(clean_birthdate2)




    # df_2 = pd.read_csv('random_sample_get_features_asCSV.csv')

    # df_2 = pd.read_csv('../../df_training_with_sim_score.csv')
    df_2 = df_5
    df_2 = df_2.drop(columns = ['Unnamed: 0.1'])

    # store the metadata to examine later with the probabilities
    original_indices = df.index
    # original_metadata = df[
    #     ['VIAF_titlelist',  'author', 'S2_titlelist', 'status', 'pub_age', 'avg_pubdate',
    #      'VIAF_birthdate', 'overlapping_words', 'word_overlap_count', 'lemma_overlap', 'overlapping_lemmas','exact_matches']].copy()

    original_metadata = df[
        ['VIAF_titlelist',  'author', 'S2_titlelist', 'status', 'pub_age', 'avg_pubdate',
         'VIAF_birthdate', 'overlapping_words', 'word_overlap_count', 'lemma_overlap', 'overlapping_lemmas','cosine_matches','sim_scores']].copy()


    df = df.drop([col for col in df.columns if col.endswith('.1')], axis=1)

    df = df.drop(columns=['overlapping_lemmas','S2_pubdates','overlapping_words','S2_titlelist','S2_Year','VIAF_titlelist','author','cosine_matches','S2_Year','birthdate','viaf_title_list','sim_scores'])  # Drop the label and metadata columns

    # df.dtypes
    print(df.dtypes)

    df_2 = df_2.drop(columns=['overlapping_lemmas','S2_pubdates','matched_title?','match?','match', 'S2_embeddings','VIAF_embeddings','overlapping_words','matched_title_list','S2_titlelist','S2_Year','VIAF_titlelist','author','cosine_matches','standard_birthdate','sim_scores'])  # Drop the label and metadata columns


    df_reordered = df[df_2.columns]

    df = df_reordered

    # df['VIAF_birthdate'] = df['VIAF_birthdate'].apply(clean_up_birthdate)

    # print(df['VIAF_birthdate'].head())
    print(df.columns)

    # df = df.drop(columns=['S2_embeddings','VIAF_embeddings'])

    # for idx, row in df.iterrows():
    #     sim_scores = row['sim_scores']
    #     num_list = []
    #     if str(sim_scores).lower() != 'nan':
    #         for x in sim_scores:
    #             x = x.strip()  # Remove any extra spaces
    #             try:
    #                 # Convert to float and add to list (skip 'NaN')
    #                 if x.lower() != 'nan':
    #                     num_list.append(float(x))
    #             except ValueError:
    #                 continue  # Skip any bad data
    #
    #         # Calculate the mean of the numbers, handle empty lists
    #         if num_list:
    #             mean_score = np.mean(num_list)  # Compute mean if list is not empty
    #         else:
    #             mean_score = 0  # Default to 0 if no valid numbers
    #
    #         # Store the mean in a new column
    #         df.at[idx, 'sim_scores_mean'] = mean_score
    #
    # df['sim_scores_mean'] = df['sim_scores_mean'].fillna(0)


    # with open('../viaf_classifier_sept23.pkl', 'rb') as file:
    #     loaded_model = pickle.load(file)
    print(loaded_model.feature_names_in_)

    # Make predictions and calculate probabilities
    predictions = loaded_model.predict(df)
    probabilities = loaded_model.predict_proba(df)  # Get class probabilities

    # Add predictions and probabilities as new columns in the DataFrame
    df['predictions'] = predictions

    # Assuming a binary classifier, add probabilities for each class
    df['probability_class_0'] = probabilities[:, 0]  # Probability of class 0
    df['probability_class_1'] = probabilities[:, 1]  # Probability of class 1

    # Combine metadata with the results
    results_df = pd.concat([original_metadata, df], axis=1)

    # Save the results
    df.to_csv('df_with_predictions.csv')
    results_df.to_csv('results_df_with_predictions.csv')
if __name__ == '__main__':
    main()