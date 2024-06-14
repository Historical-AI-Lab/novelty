# Select training pairs for training Sentence Transformer model
# this version of the script dispenses with the pairs that
# were targeted for paraphrase generation, and instead
# focuses on pairs that are either successive chunks from the
# same article.

# We attempt to make sure that the size and distribution of the
# training set is similar to the set used in the previous version.

import pandas as pd
import random, os, math, sys

import random
import math
from collections import Counter

def select_pairs(num_chunks):
    num_pairs = math.ceil(num_chunks / 8)
    
    if num_pairs == 0:
        return []
    
    # Initialize a list to mark selected chunks
    selected_pairs = []
    
    # Define the range of possible starting indices for the pairs
    available_indices = list(range(num_chunks - 1))
    
    while len(selected_pairs) < num_pairs:
        # Choose a random starting index from the available indices
        try:
            start_index = random.choice(available_indices)
        except:
            print('available_indices:', available_indices)
            print(num_chunks)
            sys.exit()
        
        # Add the pair to the list
        selected_pairs.append((start_index, start_index + 1))
        
        # Remove indices around the chosen pair to ensure separation by at least one non-selected chunk
        for i in range(start_index - 1, start_index + 3):
            if i in available_indices:
                available_indices.remove(i)
        
        # If we run out of available indices, we stop to prevent infinite loop
        if not available_indices:
            break
    
    return selected_pairs


rootfolder = "/projects/ischoolichass/ichass/usesofscale/novelty/perplexity/cleanchunks/"

meta = pd.read_csv("../metadata/litstudies/LitMetadataWithS2.tsv", sep = '\t')
meta = meta[meta['paperId'].notnull() & (meta['paperId'] != '')]

# We save the training pairs as a list of dictionaries, each with the keys
# paperId: the ID of the paper
# year: the year of the paper
# anchor: the anchor text
# positive: the positive target text
# category: the way the pair was selected

# Each year gets a list of training_pairs, and then we randomly sample N pairs per
# year. years_with_pairs is a dictionary that will store the lists for each year.

years_with_pairs = dict()

yearsizes = []

category = 'successive'

for year in range(1900, 2018):

    if year % 10 == 0:
        print(year)
    this_year = meta[meta['year'] == year]

    training_pairs = []
    
    for idx, row in this_year.iterrows():
        paper_id = row['paperId']
        path = rootfolder + paper_id + ".txt"
        if not os.path.isfile(path):
            continue
        with open(path, 'r') as f:
            textlines = f.readlines()
            textlines = [line.strip().split('\t')[1] for line in textlines]
        numberofchunks = len(textlines)

        if numberofchunks < 2:
            continue

        # Select successive chunks from the same article
        selected_pairs = select_pairs(numberofchunks)
        for pair in selected_pairs:
            anchor_chunk, positive_chunk = pair
            anchor_text = textlines[anchor_chunk]
            positive_text = textlines[positive_chunk]

            if random.random() < 0.5:
                anchor_text, positive_text = positive_text, anchor_text  # Swap anchor and positive

            this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text,
                            'category': category}
            training_pairs.append(this_pair)
    
    years_with_pairs[year] = training_pairs
    print(year, len(training_pairs))
    yearsizes.append(len(training_pairs))

training_pairs_per_year = Counter()
old_training_set = pd.read_csv("training_pairs_final_set.tsv", sep = '\t')
for idx, row in old_training_set.iterrows():
    training_pairs_per_year[row['year']] += 1

training_pairs = []
for year in years_with_pairs.keys():
    training_pairs.extend(random.sample(years_with_pairs[year], min(len(years_with_pairs[year]), training_pairs_per_year[year])))

df = pd.DataFrame(training_pairs)
df.to_csv("training_pairs_no_paraphrase.tsv", sep = '\t', index = False)
print(df.shape)    