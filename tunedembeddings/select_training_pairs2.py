# Select training pairs for training Sentence Transformer model

# We iterate through metadata, choosing 100 training pairs from each year.
# 90% of the pairs will be successive chunks from the same article,
# with the first chunk providing an anchor and the second chunk providing
# a positive target.

# 10% will be a single chunk, paired with the word "generate," which is
# a signal that we need to generate the positive target using an LLM to
# summarize and rephrase the anchor.

import pandas as pd
import random, os

rootfolder = "/projects/ischoolichass/ichass/usesofscale/novelty/perplexity/cleanchunks/"

meta = pd.read_csv("../metadata/litstudies/LitMetadataWithS2.tsv", sep = '\t')
meta = meta[meta['paperId'].notnull() & (meta['paperId'] != '')]

# Note that I'm running this script after already running an earlier version that
# selected 120 rows per year. I'm excluding those papers from the metadata so that
# we don't select the same papers again.

# In this run, I select an additional 100 rows per year. This will create a distribution
# that is flat at 220 per year from the 1930s through around 2015. It will be lower
# in the early years, because we have fewer papers from those years.

training_pairs_df = pd.read_csv("training_pairs.tsv", sep='\t')
paper_ids = set(training_pairs_df['paperId'])
meta = meta[~meta['paperId'].isin(paper_ids)]

# We save the training pairs as a list of dictionaries, each with the keys
# paperId: the ID of the paper
# year: the year of the paper
# anchor: the anchor text
# positive: the positive target text

training_pairs = []

for year in range(1900, 2018):

    if year % 10 == 0:
        print(year)
    this_year = meta[meta['year'] == year]
    n = 100
    if len(this_year) < 100:
        n = len(this_year)  # If there are fewer than 100 articles in a year, use all of them
    this_year = this_year.sample(n)
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

        if random.random() < 0.9:
            # Select successive chunks from the same article
            anchor_chunk = random.randint(0, numberofchunks - 2)
            positive_chunk = anchor_chunk + 1
            anchor_text = textlines[anchor_chunk]
            positive_text = textlines[positive_chunk]

            # We don't want to memorize the order of the anchor and positive
            # and learn an entailment model. So we randomly swap the order.

            if random.random() < 0.5:
                anchor_text, positive_text = positive_text, anchor_text  # Swap anchor and positive
            this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text}
            training_pairs.append(this_pair)

        else:
            # Select a single chunk paired with the word "generate"
            chunk_index = random.randint(0, numberofchunks - 1)
            anchor_text = textlines[chunk_index]
            positive_text = "generate"
            
            if random.random() < 0.5:
                anchor_text, positive_text = positive_text, anchor_text  # Swap anchor and positive
            
            # We swap 30% of the generated texts, to ensure that the model
            # has some examples of LLM style in the anchor column. This will
            # make it harder for the model to match texts based on style alone.

            this_pair = {'paperId': paper_id, 'year': year, 'anchor': anchor_text, 'positive': positive_text}
            training_pairs.append(this_pair)

df = pd.DataFrame(training_pairs)
df.to_csv("training_pairs_set2.tsv", sep = '\t', index = False)