# Apply Sentence-Bert Model

from sentence_transformers import SentenceTransformer
import pandas as pd
import os, sys
import argparse

# The command line arguments include
# -m --modelpath  the path to the Sentence Transformer model
# -s --startyear  the start year for embeddings (must be a multiple of 10)
# -e --endyear    the end year for embeddings (must be a multiple of 10)
# -o --outputpath the path to the output directory

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Apply Sentence-Bert Model')
parser.add_argument('-m', '--modelpath', type=str, help='the path to the Sentence Transformer model')
parser.add_argument('-s', '--startyear', type=int, help='the start year for embeddings (must be a multiple of 10)')
parser.add_argument('-e', '--endyear', type=int, help='the end year for embeddings (must be a multiple of 10)')
parser.add_argument('-o', '--outputpath', type=str, help='the path to the output directory, should not end with slash')
args = parser.parse_args()

# Extract command-line arguments
modelpath = args.modelpath
startyear = args.startyear
endyear = args.endyear
outputpath = args.outputpath

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer('models/run_60000pairs/checkpoint-4175')

meta = pd.read_csv("../metadata/litstudies/LitMetadataWithS2.tsv", sep = '\t')
meta = meta[meta['paperId'].notnull() & (meta['paperId'] != '')]

rootfolder = "../perplexity/cleanchunks/"

print('metadata loaded')

for decade in range(start, end, 10):
    print(decade, flush = True)
    paperIds = []
    chunknumbers = []
    paragraphs = []
    this_decade = meta[(meta['year'] >= decade) & (meta['year'] < decade + 10)]
    for idx, row in this_decade.iterrows():
        paper_id = row['paperId']
        path = rootfolder + paper_id + ".txt"
        if not os.path.isfile(path):
            continue
        with open(path, 'r') as f:
            textlines = f.readlines()
            textlines = [line.strip().split('\t')[1] for line in textlines]
        numberofchunks = len(textlines)
        paperIds.extend([paper_id] * numberofchunks)
        chunknumbers.extend(list(range(numberofchunks)))
        paragraphs.extend(textlines)
    
    print('paragraphs loaded', flush = True)
    
    embeddings = model.encode(paragraphs, batch_size = 128, device = 'cuda')
    numrows, numcols = embeddings.shape
    # Create a dataframe with paperIds, chunknumbers, and embeddings
    data = {'paperId': paperIds, 'chunknumber': chunknumbers}
    for i in range(numcols):
        data[f'embedding_{i+1}'] = embeddings[:, i]

    df = pd.DataFrame(data)

    outpath = outputpath + "/" + str(decade) + ".tsv"
    df.to_csv(outpath, sep = '\t', index = False)
    