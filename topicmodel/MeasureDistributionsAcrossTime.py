# Calculate distributions for topic model datasets

# Our goal here is to figure out how many chunks and words per year
# we have both in the original dataset (litstudiesforLDA.txt) and
# in the resampled one (ResampledLitStudiesForLDA.txt).

import sys
import pandas as pd
from collections import Counter

meta = pd.read_csv('../metadata/litstudies/LitMetadataWithS2.tsv', sep='\t')
meta = meta[~meta['paperId'].isnull() & (meta['paperId'] != '')]

chunksperyear = Counter()
wordsperyear = Counter()

# we store data as chunkID -> text key-value pairs
# at the same time
errors = 0

with open('../embeddingcode/litstudiesforLDA.txt', 'r') as f:
    for line in f:
        try:
            chunkID, label, text = line.strip().split('\t')
            wordcount = len(text.split())
        except:
            errors += 1
            if errors % 10 == 1:
                print('Errors:', errors)
        paperId = chunkID.split('-')[0]
        year = int(meta.loc[meta['paperId'] == paperId, 'year'].values[0])
        chunksperyear[year] += 1
        wordsperyear[year] += wordcount


print('Errors in original: ', errors)
errors = 0
resampledchunks = Counter()
resampledwords = Counter()

with open('ResampledLitStudiesForLDA.txt', 'r') as f:
    for line in f:
        try:
            chunkID, label, text = line.strip().split('\t')
            wordcount = len(text.split())
        except:
            errors += 1
            if errors % 10 == 1:
                print('Errors:', errors)
        paperId = chunkID.split('-')[0]
        year = int(meta.loc[meta['paperId'] == paperId, 'year'].values[0])
        resampledchunks[year] += 1
        resampledwords[year] += wordcount

print('Errors in resampled: ', errors)

# Create a dataframe with the desired columns
data = {'year': [], 'originalchunkct': [], 'originalwordct': [], 'resampledchunkct': [], 'resampledwordct': []}

# Iterate over the years
for year in sorted(set(chunksperyear.keys()) | set(resampledchunks.keys())):
    # Get the counts for the original dataset
    original_chunk_count = chunksperyear.get(year, 0)
    original_word_count = wordsperyear.get(year, 0)
    
    # Get the counts for the resampled dataset
    resampled_chunk_count = resampledchunks.get(year, 0)
    resampled_word_count = resampledwords.get(year, 0)
    
    # Append the data to the dataframe
    data['year'].append(year)
    data['originalchunkct'].append(original_chunk_count)
    data['originalwordct'].append(original_word_count)
    data['resampledchunkct'].append(resampled_chunk_count)
    data['resampledwordct'].append(resampled_word_count)

# Create the dataframe
df = pd.DataFrame(data)
df.to_csv('LitStudiesDatasetSizeByYear.tsv', sep = '\t', index = False)