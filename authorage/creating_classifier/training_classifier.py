"""
train classifier on known sample?
"""

#import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('result_df_replicatedwembeddings.csv')


df.drop(['Unnamed: 0','match?','S2titles','record_enumerated_titles','title_list','common_words','S2_Author','word_overlap','match','S2Titles','matched_title?','matched_title_list','notes','S2_pubdates','VIAF_embeddings','S2_embeddings','record_count','birthdate','author'], axis=1, inplace=True)
# print(df.columns)
# X = df.drop('selected_birthyear')
# df = df.loc[df[['avg_pubdate','VIAF_birthdate','standard_birthdate','author','average_S2_pubdate']].dropna()]
df = df.dropna(subset=['avg_pubdate', 'VIAF_birthdate', 'standard_birthdate', 'average_S2_pubdate'])


print(df.columns)

Y = df['selected_birthyear']
X = df.drop(['selected_birthyear'],axis=1)
print(df.columns)

print(df[df.isnull().any(axis=1)])
print(df.isnull().values.any())

#
# #
# #Create classifier
# regr_1 = DecisionTreeRegressor(max_depth=2)
# #regr_1 = RandomForestRegressor(n_estimators=5)
# regr_1.fit(X,Y)
# Y_pred=regr_1.predict(X)
#
# r2_score(Y, Y_pred)
#
# print('Accuracy: ',r2_score(Y, Y_pred))
#
# importances = list(regr_1.feature_importances_)
# #Print out the feature and importances
# print (importances)
#
# confusion_matrix = pd.crosstab(Y, Y_pred, rownames=['Actual'], colnames=['Predicted'])
# sns.heatmap(confusion_matrix, annot=True)