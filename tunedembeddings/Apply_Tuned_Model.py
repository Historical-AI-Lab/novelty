# Apply Sentence-Bert Model

from sentence_transformers import SentenceTransformer
import pandas as pd
import os, sys

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer('models/final_60000pairs/checkpoint-4175')

meta = pd.read_csv("../metadata/litstudies/LitMetadataWithS2.tsv", sep = '\t')
meta = meta[meta['paperId'].notnull() & (meta['paperId'] != '')]

rootfolder = "../perplexity/cleanchunks/"

print('metadata loaded')

args = sys.argv
if len(args) > 2:
    start = int(args[1])
    end = int(args[2])
else:
    print('Please provide start and end years for the embeddings. Must be multiples of 10.')
    sys.exit(0)

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

    outpath = "finalembeds/" + str(decade) + ".tsv"
    df.to_csv(outpath, sep = '\t', index = False)
    





        


# The sentences to encode
sentences = [
    "The weather is lovely today.",
    "It's so sunny outside!",
    "He drove to the stadium.",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)