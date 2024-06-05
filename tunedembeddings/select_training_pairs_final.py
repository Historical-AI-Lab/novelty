# Select training pairs for training Sentence Transformer model

# We iterate through metadata, choosing 100 training pairs from each year.
# 90% of the pairs will be successive chunks from the same article,
# with the first chunk providing an anchor and the second chunk providing
# a positive target.

# 10% will be a single chunk, paired with the word "generate," which is
# a signal that we need to generate the positive target using an LLM to
# summarize and rephrase the anchor.

import pandas as pd
import random, os, math, sys

import random
import math

args = sys.argv

N = args[1]   # number of pairs to select per year
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
        start_index = random.choice(available_indices)
        
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

completion_categories = Counter()

# The possible categories are
# single-synthetic: a single chunk paired with the word "generate," which is a signal to generate a
#                   paraphrase of the chunk using an LLM. 8% of all pairs.
# successive: two successive chunks from the same article. 88% of all pairs.
# random-pair: A random pair of chunks from the same article, not necessarily successive.
#              articles used this way cannot be used in any other category, to prevent overlap
#              Since these may be hard to match, only 2% of all pairs.
# synthetic-pair: a successive pair of chunks from the same article, of which one should be replaced
#                 by a synthetic paraphrase. Only 2% of all pairs.

yearsizes = []

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

        category_roll = random.random()

        if category_roll < 0.98 or numberofchunks > 20:
            # Select successive chunks from the same article
            selected_pairs = select_pairs(numberofchunks)
            for pair in selected_pairs:
                anchor_chunk, positive_chunk = pair
                anchor_text = textlines[anchor_chunk]
                positive_text = textlines[positive_chunk]
                category = random.choices(['successive', 'synthetic-pair', 'single-synthetic'], weights=[0.91, 0.02, 0.07])[0]
                if category == 'single-synthetic':
                    positive_text = 'generate'

                if random.random() < 0.5:
                    anchor_text, positive_text = positive_text, anchor_text  # Swap anchor and positive

                this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text,
                              'category': category}
                training_pairs.append(this_pair)

        else:
            # Select a random pair of chunks from the same article
            anchor_chunk = 0
            positive_chunk = 0
            while anchor_chunk == positive_chunk:
                anchor_chunk = random.randint(0, numberofchunks - 1)
                positive_chunk = random.randint(0, numberofchunks - 1)
            anchor_text = textlines[anchor_chunk]
            positive_text = textlines[positive_chunk]
            
            this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text,
                         'category': 'random-pair'}
            training_pairs.append(this_pair)
    
    years_with_pairs[year] = training_pairs
    print(year, len(training_pairs))
    yearsizes.append(len(training_pairs))

number_of_yearly_pairs = sorted(yearsizes)
maxsize = min(N, number_of_yearly_pairs[25])
print(f'Capping the number of pairs per year at {maxsize}')

training_pairs = []
for year in years_with_pairs.keys():
    training_pairs.extend(random.sample(years_with_pairs[year], min(len(years_with_pairs[year]), maxsize)))

df = pd.DataFrame(training_pairs)
df.to_csv("training_pairs_final_set.tsv", sep = '\t', index = False)