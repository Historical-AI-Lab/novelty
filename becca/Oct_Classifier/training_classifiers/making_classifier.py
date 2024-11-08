# from random_sample_for_training import clean_and_org_randomsample_claude.py
#comment
import random_sample_for_training
import all_data

from random_sample_for_training import clean_and_org_randomsample_claude
from random_sample_for_training import random_sample_get_features
from random_sample_for_training import train_model_onsample

from all_data import clean_and_org_claude_all_data
from all_data import getting_the_features
from all_data import getting_the_features_2
from all_data import run_the_model

import pandas as pd
import json
import pickle

meta = pd.read_csv('../../../data_sources/semantic_scholar/LitStudiesMetaWithS2.tsv', sep='\t')
df_july = pd.read_csv('result_df_july18_1136am.csv')

with open('S2_data_dict.txt', 'r') as filename:
    S2_data_dict = json.load(filename)

with open('../viaf_classifier_sept23.pkl', 'rb') as model:
    loaded_model = pickle.load(model)
#this file is the resulting model trained by the random_sample modules

df = pd.read_csv('processed_search_results_VIAF_S2_Oct.csv')
#this file is created by running the module clean_and_organize_randomsample_to_train.py

df_2 = pd.read_csv('../new_labels_aug22.csv')
#this file contains corrections to ground truth labels based on re-examination

df_3 = pd.read_csv('../random_sample_get_features_asCSV.csv')
#this file comes from running random_sample_get_features.py

df_4 = pd.read_csv('../search_results_3.csv')
#VIAF search results for all data? -- find source of this file originally/code tha tproduced it
    #file is created in clean_and_combine_data_b4_features.py

df_5 = pd.read_csv('../df_training_with_sim_score.csv')
#from train_model_onsample

df_6 = pd.read_csv('../all_data_get_features_asCSV.csv')
#from clean_and_org_claude_all_data


# Run the main portion of each imported script

#####
#Train the model
#####

random_sample_for_training.clean_and_org_randomsample_claude.main(meta, df_july)
random_sample_for_training.random_sample_get_features.main(S2_data_dict,df)
random_sample_for_training.train_model_onsample.main(df_2,df_3)


#######
#Run the model over all data
######

all_data.clean_and_org_claude_all_data.main(meta, df_4)
all_data.getting_the_features.main(df_4, S2_data_dict)
#check what is difference between getting the features and getting the features 2?
    #I think just that bugs were fixed in the 2 version and not in the first version of it?
all_data.run_the_model.main(df_5, df_6, loaded_model)