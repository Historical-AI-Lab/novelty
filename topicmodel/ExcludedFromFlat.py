# Make data for chunks excluded from flat sample

# We read data used in the flat sample, record chunk Ids, 
# and then go through the whole dataset, excluding chunks that are in the flat sample.

import sys
import pandas as pd
import random

chunksinflat = set()

errors = 0

with open('ResampledLitStudiesForLDA.txt', 'r') as f:
    for line in f:
        try:
            chunkID, label, text = line.strip().split('\t')
        except:
            errors += 1
            if errors % 10 == 1:
                print('Errors:', errors)
        chunksinflat.add(chunkID)

outlines = []

with open('../embeddingcode/litstudiesforLDA.txt', 'r') as f:
    for line in f:
        try:
            chunkID, label, text = line.strip().split('\t')
        except:
            errors += 1
            if errors % 10 == 1:
                print('Errors:', errors)
        if chunkID in chunksinflat:
            continue
        else:
            outlines.append(line)

print('Errors:', errors)

with open('ExcludedFromFlat.txt', mode = 'w', encoding = 'utf-8') as f:
    for outline in outlines:
        f.write(outline)