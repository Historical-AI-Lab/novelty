import pandas as pd

df = pd.read_csv('../random_sample_get_features_asCSV.csv')
print(df.columns)

# df = df.drop('birthyear', axis=1)
print(df.head(30))


# import pandas as pd
# from sklearn.model_selection import train_test_split, cross_val_predict
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (precision_recall_curve, roc_auc_score,
#                              confusion_matrix, ConfusionMatrixDisplay,
#                              classification_report, accuracy_score)
# import matplotlib.pyplot as plt
#
#
# X = df.drop(columns=['overlapping_lemmas','title_list','record_enumerated_titles','S2Titles','S2titles','matched_title_list','common_words','notes','normalized_author','S2_Titlelist','S2_Author','S2_pubdates','mean_embedding','matched_title?','match?','match'])  # Drop the label and metadata columns
# y = df['match?']
#
# # Optional: If you want to split into training and test sets first
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42
# )
#
# # Initialize the Random Forest model
# model = RandomForestClassifier(n_estimators=100, random_state=42)
#
# # Perform cross-validation
# model = RandomForestClassifier()
# y_pred_proba = cross_val_predict(model, X, y, cv=5, method='predict_proba')
#
# # Store predictions with original indices
# predictions_df = pd.DataFrame(y_pred_proba, index=original_indices, columns=['Class_0_Proba', 'Class_1_Proba'])
#
# # Merge the metadata back
# results_df = pd.concat([original_metadata, predictions_df], axis=1)
# results_df['True_Label'] = y.loc[results_df.index]
#
# y_pred = (y_pred_proba[:, 1] >= 0.9).astype(int)
#
# # Now you have a DataFrame with metadata and predictions
# print(results_df.head())
#
# # Generate classification report based on the true labels (y_train) and predicted labels (y_pred)
# report = classification_report(y, y_pred)
#
# # Print the classification report
# print(report)
#
# import pickle
#
# model.fit(X, y)
# with open('viaf_classifier_sept23.pkl', 'wb') as file:
#     pickle.dump(model, file)