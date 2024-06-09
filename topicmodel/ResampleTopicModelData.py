# Resample topic model data

# We read data used for a previous topic model, identify the date of each line,
# and resample the data so it has a more even distribution of dates.

import sys
import pandas as pd
import random

meta = pd.read_csv('../metadata/litstudies/LitMetadataWithS2.tsv', sep='\t')
meta = meta[~meta['paperId'].isnull() & (meta['paperId'] != '')]

data = dict()
chunksperyear = dict()

# we store data as chunkID -> text key-value pairs
# at the same time
errors = 0

with open('../embeddingcode/litstudiesforLDA.txt', 'r') as f:
    for line in f:
        try:
            chunkID, label, text = line.strip().split('\t')
        except:
            errors += 1
            if errors % 10 == 1:
                print('Errors:', errors)
        data[chunkID] = text
        paperId = chunkID.split('-')[0]
        year = int(meta.loc[meta['paperId'] == paperId, 'year'].values[0])
        if year not in chunksperyear:
            chunksperyear[year] = []
        chunksperyear[year].append(chunkID)

for yr in range(1900, 2019):
    if yr in chunksperyear:
        print(yr, len(chunksperyear[yr]))

print()
print('Total chunks:', len(data))
print()

# Create a list of the number of chunks in each year

chunkcounts = [len(chunksperyear[x]) for x in chunksperyear.keys()]
chunkcounts.sort()
cap = chunkcounts[40]   # this is the number of chunks in the year 40th from the bottom
print(f'Years will be capped at {cap} chunks.')
print('Resampling...')
print()

count = 0
with open('ResampledLitStudiesForLDA.txt', mode = 'w', encoding = 'utf-8') as f:
    for yr in chunksperyear:
        if len(chunksperyear[yr]) > cap:
            sample = random.sample(chunksperyear[yr], cap)
        else:
            sample = chunksperyear[yr]
        for chunkID in sample:
            f.write(chunkID + '\t' + 'l1' + '\t' + data[chunkID] + '\n')
            count += 1

print('Resampling complete.')
print('Total chunks:', count)




        

