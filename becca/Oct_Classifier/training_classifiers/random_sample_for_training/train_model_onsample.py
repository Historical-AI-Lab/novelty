
def main(df_2,df_3):
    import pandas as pd
    df = df_3
    # df = pd.read_csv('../../random_sample_get_features_asCSV.csv')
    print(df.columns)
    # columns = ['Unnamed: 0']
    # df = df.drop(columns)
    #in this case, I had done the true labels by hand, so let's add them in here
    # df_2 = pd.read_csv('../../new_labels_aug22.csv')
    df['match?'] = df_2['new_label']

    # c = ['S2_pubdates','author','overlapping_words','overlapping_lemmas','exact_matches']
    # c = ['birthyear','Unnamed: 0']
    # df = df.drop(c, axis=1)

    # print(df.head(50))

    # df_notnull_birth2max = df.loc[df['birth2maxdate'].notnull()]
    # print(df_notnull_birth2max.head(50))


    import pandas as pd
    from sklearn.model_selection import train_test_split, cross_val_predict
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (precision_recall_curve, roc_auc_score,
                                 confusion_matrix, ConfusionMatrixDisplay,
                                 classification_report, accuracy_score)
    import matplotlib.pyplot as plt
    #

    # store the metadata to examine later with the probabilities
    original_indices = df.index
    # original_metadata = df[
    #     ['VIAF_titlelist',  'author', 'S2_titlelist', 'status', 'pub_age', 'avg_pubdate',
    #      'VIAF_birthdate', 'overlapping_words', 'word_overlap_count', 'lemma_overlap', 'overlapping_lemmas','exact_matches']].copy()

    original_metadata = df[
        ['VIAF_titlelist',  'author', 'S2_titlelist', 'status', 'pub_age', 'avg_pubdate',
         'VIAF_birthdate', 'overlapping_words', 'word_overlap_count', 'lemma_overlap', 'overlapping_lemmas','cosine_matches','sim_scores']].copy()
    print(df)
    #
    X = df.drop(columns=['overlapping_lemmas','S2_pubdates','matched_title?','match?','match', 'S2_embeddings','VIAF_embeddings','overlapping_words','matched_title_list','S2_titlelist','S2_Year','VIAF_titlelist','author','cosine_matches','standard_birthdate'])  # Drop the label and metadata columns
    y = df['match?']

    print(X.dtypes)

    # X = X.astype(float)
    import numpy as np
    import ast

    # df['sim_scores'] = [int(x) if not np.isnan(x) else '0' for x in (ast.literal_eval(df['sim_scores']))]  # Replace NaN with 0
    # Apply ast.literal_eval row by row and handle NaN conversion
    # df['sim_scores'] = df['sim_scores'].apply(
    #     lambda x: [int(val) if not np.isnan(val) else 0 for val in ast.literal_eval(x)]
    #     if isinstance(x, str) else [0])  # If x is not a string, default to [0]
    # # df['sim_scores'] = [int(x) for x in df['sim_scores']]

    # def process_sim_scores(val):
    #     if isinstance(val, str):  # If it's a string, try to evaluate it as a list
    #         try:
    #             val = ast.literal_eval(val)
    #         except (ValueError, SyntaxError):  # In case the string isn't a valid list
    #             return [0]
    #     if isinstance(val, list):  # If it's already a list, process it
    #         return [int(x) if not np.isnan(x) else [0] for x in val]
    #     elif np.isnan(val):  # If it's a NaN
    #         return [0]
    #     else:  # If it's a single number (int or float)
    #         return [int(val)] if not np.isnan(val) else [0]

    # Apply the function to the 'sim_scores' column
    # df['sim_scores'] = df['sim_scores'].apply(process_sim_scores)

    X = df.drop(columns=['overlapping_lemmas','S2_pubdates','matched_title?','match?','match', 'S2_embeddings','VIAF_embeddings','overlapping_words','matched_title_list','S2_titlelist','S2_Year','VIAF_titlelist','author','cosine_matches','standard_birthdate'])  # Drop the label and metadata columns
    y = df['match?']

    # # Expand sim_scores lists into separate columns
    # sim_scores_expanded = pd.DataFrame(df['sim_scores'].tolist(), index=df.index)
    #
    # # Combine expanded sim_scores back with the original dataframe
    # df = pd.concat([df.drop(columns=['sim_scores']), sim_scores_expanded], axis=1)


    # for idx, row in df.iterrows():
    #     sim_scores = str(row['sim_scores'])
    #     sim_scores = sim_scores.strip('[').strip(']')
    #     sim_scores = sim_scores.split(',')
    #     int_list = [int(float(x)) if str(x).lower() != 'nan' and not np.isnan(float(x)) else 0 for x in str(sim_scores)]

    import numpy as np

    # for idx, row in df.iterrows():
    #     sim_scores = row['sim_scores']
    #
    #     # Only process if the value is a string
    #     if isinstance(sim_scores, str):
    #         # Remove the square brackets
    #         sim_scores = sim_scores.strip('[').strip(']')
    #
    #         # Split the string by commas to get individual elements
    #         sim_scores = sim_scores.split(',')
    #
    #         # Convert the elements into floats and handle 'NaN' manually
    #         int_list = []
    #         for x in sim_scores:
    #             x = x.strip()  # Remove any extra spaces
    #             if x.lower() == 'nan':  # Handle 'NaN' strings
    #                 int_list.append(0)
    #             else:
    #                 try:
    #                     int_list.append(int(float(x)))  # Convert to int after float conversion
    #                 except ValueError:
    #                     int_list.append(0)  # Handle any bad data (e.g., empty strings)
    #
    #         # Store processed results in a new column
    #         df.at[idx, 'sim_scores_processed'] = str(int_list)

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

    df.to_csv('df_training_with_sim_score.csv')




    X = df.drop(columns=['overlapping_lemmas','S2_pubdates','matched_title?','match?','match', 'S2_embeddings','VIAF_embeddings','overlapping_words','matched_title_list','S2_titlelist','S2_Year','VIAF_titlelist','author','cosine_matches','standard_birthdate','sim_scores'])  # Drop the label and metadata columns
    y = df['match?']

    print(X.dtypes)


    #
    # # Optional: If you want to split into training and test sets first
    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, y, test_size=0.2, random_state=42
    # )
    #
    # # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    #



    y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')
    #
    # # Store predictions with original indices
    predictions_df = pd.DataFrame(y_pred_proba, index=original_indices, columns=['Class_0_Proba', 'Class_1_Proba'])
    #
    # # Merge the metadata back
    results_df = pd.concat([original_metadata, predictions_df], axis=1)
    results_df['True_Label'] = y.loc[results_df.index]
    #
    y_pred = (y_pred_proba[:, 1] >= 0.9).astype(int)
    #
    # # Now you have a DataFrame with metadata and predictions
    print(results_df.head())

    results_df.to_csv('df_training_with_label_proba.csv')
    #
    # # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
    report = classification_report(y, y_pred)
    #
    # # Print the classification report
    print(report)
    #
    import pickle

    model.fit(X, y)
    with open('../viaf_classifier_sept23.pkl', 'wb') as file:
        pickle.dump(model, file)
if __name__ == '__main__':
    main()