# Select training pairs for training Sentence Transformer model
# This script was used to create training data for fiction.

# We iterate through metadata, selecting files.
# For each file we select pairs of chunks separated
# by one non-selected "spacer" chunk. There is a 5% chance that
# any given non-selected chunk will be used as a single-synthetic
# pair, where GPT-3.5 is asked to generate a paraphrase of the chunk.
# There is also a 5% chance that a successive pair will be used as a
# synthetic pair, where one of the chunks is replaced by a synthetic
# paraphrase. The remaining 90% of pairs are successive pairs.

import pandas as pd
import random, os, math, sys

import random
import math
from collections import Counter

# Print current working directory
print(os.getcwd())

# args = sys.argv

# N = int(args[1])   # number of pairs to select per year
# we will use the minimum of this number and the 25th smallest year

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
        
        # Remove indices around the chosen pair to ensure separation by at least two non-selected chunks
        for i in range(start_index - 2, start_index + 4):
            if i in available_indices:
                available_indices.remove(i)
        
        # If we run out of available indices, we stop to prevent infinite loop
        if not available_indices:
            break
    
    return selected_pairs


rootfolder = "novelty/fiction/fiction_chunks/"

meta = pd.read_csv("novelty/fiction/capped_fiction_metadata.tsv", sep = '\t')

# We save the training pairs as a list of dictionaries, each with the keys
# docid: the document id of the book
# year: the year of the book
# anchor: the anchor text
# positive: the positive target text
# category: the way the pair was selected

# Each year gets a list of training_pairs, and then we randomly sample N pairs per
# year. years_with_pairs is a dictionary that will store the lists for each year.

years_with_pairs = dict()

completion_categories = Counter()

# The possible categories are
# single-synthetic: a single chunk paired with the word "generate," which is a signal to generate a
#                   paraphrase of the chunk using an LLM. 5% of all pairs.
# successive: two successive chunks from the same article. 90% of all pairs.
# synthetic-pair: a successive pair of chunks from the same article, of which one should be replaced
#                 by a synthetic paraphrase. Only 5% of all pairs.

yearsizes = dict()

for year in range(1890, 2000):

    if year % 10 == 0:
        print(year)
    this_year = meta[meta['PUBL_DATE'] == year]
    this_year = this_year[this_year['SOURCE'] != 'extracted_features']

    training_pairs = []
    
    for idx, row in this_year.iterrows():
        paper_id = row['BOOK_ID']
        filename = row['FILENAME']
        path = rootfolder + filename
        if not os.path.isfile(path):
            print('File not found:', path)
            continue
        with open(path, 'r') as f:
            textlines = f.readlines()
            textlines = [line.strip().split('\t')[1] for line in textlines]
        numberofchunks = len(textlines)

        if numberofchunks < 1:
            continue
        elif numberofchunks == 1:    # we can only use single-chunk files for synthetic completion
            anchor_text = textlines[0]
            positive_text = "generate"
            if random.random() < 0.5:
                anchor_text, positive_text = positive_text, anchor_text
            category = "single-synthetic"
            this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text, 'category': category}
            training_pairs.append(this_pair)
            continue

        else:
            for i in range(0, numberofchunks, 3):
                if i + 1 < numberofchunks:
                    anchor_text = textlines[i]
                    positive_text = textlines[i + 1]
                    if random.random() < 0.05:
                        category = "synthetic-pair"
                    else:
                        category = "successive"
                    if random.random() < 0.5:
                        anchor_text, positive_text = positive_text, anchor_text
                    this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text, 'category': category}
                    training_pairs.append(this_pair)
                    if i + 2 < numberofchunks and random.random() < 0.05:
                        anchor_text = textlines[i + 2]
                        positive_text = "generate"
                        category = "single-synthetic"
                        if random.random() < 0.5:
                            anchor_text, positive_text = positive_text, anchor_text
                        this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text, 'category': category}
                        training_pairs.append(this_pair)
    
    years_with_pairs[year] = training_pairs
    print(year, len(training_pairs))
    yearsizes[year] = len(training_pairs)

number_of_yearly_pairs = sorted(yearsizes.values())
maxsize = number_of_yearly_pairs[5]
print(f'Capping the number of pairs per year at {maxsize}')

training_pairs = []
for year in years_with_pairs.keys():
    training_pairs.extend(random.sample(years_with_pairs[year], min(yearsizes[year], maxsize)))
    yearsizes[year] = min(yearsizes[year], maxsize)

df = pd.DataFrame(training_pairs)
df.to_csv("novelty/fiction/training_pairs_fiction.tsv", sep = '\t', index = False)
print(df.shape)    

with open('novelty/fiction/pairs_by_year.txt', 'w') as f:
    for year in yearsizes.keys():
        f.write(str(year) + '\t' + str(yearsizes[year]) + '\n')