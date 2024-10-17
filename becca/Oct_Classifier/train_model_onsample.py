import pandas as pd

df = pd.read_csv('random_sample_get_features_asCSV.csv')
print(df.columns)
# columns = ['Unnamed: 0']
# df = df.drop(columns)
#in this case, I had done the true labels by hand, so let's add them in here
df_2 = pd.read_csv('new_labels_aug22.csv')
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
     'VIAF_birthdate', 'overlapping_words', 'word_overlap_count', 'lemma_overlap', 'overlapping_lemmas','cosine_matches']].copy()
print(df)
#
X = df.drop(columns=['overlapping_lemmas','S2_pubdates','matched_title?','match?','match', 'S2_embeddings','VIAF_embeddings','overlapping_words','matched_title_list','S2_titlelist','S2_Year','VIAF_titlelist','author','cosine_matches'])  # Drop the label and metadata columns
y = df['match?']

print(X.dtypes)

X = X.astype(float)
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
#
# # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
report = classification_report(y, y_pred)
#
# # Print the classification report
print(report)
#
import pickle

model.fit(X, y)
with open('viaf_classifier_sept23.pkl', 'wb') as file:
    pickle.dump(model, file)