# Calculate precocity for RoBERTa embeddings

# This is based on calculate_precocity,
# but adapted in 2024 to process the RoBERTa
# embeddings. The main significant change is that
# the script takes a list of articles the regression
# head was trained on, which can optionally be
# excluded from calculation.

# We also distinguish chunks that are excluded from processing because
# they're by the same authors from those that are excluded because
# one chunk cites or quotes the other.

# Overall the "filtered" variable has four possible states:
# nofilter -- no filtering at all everything included
# train -- only training corpus excluded
# trainauth -- training corpus and author overlaps excluded
# trainauthquote -- training corpus, author overlaps, and cites/quotes excluded

# We don't create the sets of excluded chunks for those states here;
# they're created in the worker script. However, we do create a dictionary
# called "exclusions" that the worker script will use to determine
# which chunks to exclude. It includes the set of articles used in training
# under the key 'train' along with a sub-dictionary for each paperId
# that includes 'author overlaps' and 'chunks that quote.'
# The 'author overlaps' set is a set of paperIds that share authors with
# the center document. The 'chunks that quote' set is a set of chunkIds
# that cite or quote the center document. These will be used in the
# worker script to create sets with a matryoshka doll structure, where
# the inner set is the articles in 'train'. That's enclosed by a larger
# set (trainauth) that includes the 'author overlaps.' Finally,
# the largest set (trainauthquote) also includes the 'chunks that quote.'

# The script is designed to be run in parallel on a cluster, with
# each process calculating precocity for a different year.

# USAGE

# python calculate_prec.py metapath datapath excludepath startdate enddate

# where

# startdate is inclusive
# enddate exclusive


import pandas as pd
import numpy as np
import random, sys, os
from multiprocessing import Pool
import calc_precocity_worker_for_tuned as cpw
from ast import literal_eval
from collections import Counter

metapath = sys.argv[1]
datafolder = sys.argv[2]
chunk_level_exclude = sys.argv[3]
startdate = int(sys.argv[4])
enddate = int(sys.argv[5])
article_level_exclude = sys.argv[7]

def get_metadata(filepath):
    '''
    Loads the metadata spreadsheet and applies literal_eval to the
    authors column.
    '''
    meta = pd.read_csv(filepath, sep = '\t')
    meta['authors'] = meta['authors'].apply(literal_eval)
    
    return meta

meta = get_metadata(metapath)  # converts the author strings to lists
data = dict()
exclusions = dict()
files_in_training_set = set()

# there is no training set for this version of the script;
# the model we're testing was trained in an unsupervised way

exclusions['train'] = set()  # empty set
print('There are ', len(exclusions['train']), ' articles to be excluded because they were used in training.')
# then we load the article-level exclusions

with open(article_level_exclude, encoding= 'utf-8') as f:
    for line in f:
        fields = line.strip().split('\t')
        centerdoc = fields[0]
        whattoexclude = fields[1]
        if centerdoc not in exclusions:
            exclusions[centerdoc] = dict()
            exclusions[centerdoc]['author overlaps'] = set() # this will be a set of article ids
            exclusions[centerdoc]['chunks that quote'] = set()   # this will be a set of chunk ids

        exclusions[centerdoc]['author overlaps'].add(whattoexclude)

# Then we load the needed data files. 
# In this version of the script there is only one file

with open(datapath, encoding = "utf-8") as f:
    for line in f:
        fields = line.strip().split('\t')
        if fields[0] == 'paperId':
            continue
        paperid = fields[0]
        chunkdigits = fields[1]  # this will be a string
        chunkid = paperid + '-' + chunkdigits
        vector = np.array([float(x) for x in fields[3:]], dtype = np.float32)
        data[chunkid] = vector

# filelist = os.listdir(datafolder)
# for filename in filelist:
#     if filename.startswith('regembeddings'):
#         fields = filename.replace('regembeddings', '').split('.')[0].split('-')
#         start = int(fields[0])
#         end = int(fields[1])
#         if end >= startdate - 20 or start <= enddate + 20:
#             datapath = os.path.join(datafolder, filename)
#             with open(datapath, encoding = "utf-8") as f:
#                 for line in f:
#                     fields = line.strip().split('\t')
#                     if fields[0] == 'paperId':
#                         continue
#                     paperid = fields[0]
#                     chunkcounter[paperid] += 1
#                     chunkid = paperid + '-' + str(chunkcounter[paperid] - 1)
#                     vector = np.array([float(x) for x in fields[3:]], dtype = np.float64)
#                     data[chunkid] = vector

print('Data loaded.')

# Now we load the chunk-level exclusions

with open(chunk_level_exclude, encoding = "utf-8") as f:
    for line in f:
        fields = line.strip().split('\t')
        centerdoc = fields[0]
        if centerdoc not in exclusions:
            exclusions[centerdoc] = dict()
            exclusions[centerdoc]['author overlaps'] = set() # this will be a set of article ids
            exclusions[centerdoc]['chunks that quote'] = set()   # this will be a set of chunk ids

        for field in fields[1:]:
            exclusions[centerdoc]['chunks that quote'].add(field)
        
totalvols = meta.shape[0]

spanstocalculate = []
for centerdate in range(startdate, enddate):
    df = meta.loc[(meta.year >= centerdate - 20) & (meta.year <= centerdate + 20) &
    (~pd.isnull(meta.paperId)), : ]
    df.set_index('paperId', inplace = True)
    spanstocalculate.append((centerdate, df))

outputname = 'precocity_tuned_' + str(startdate)
summaryfile = 'results/' + outputname + 's_docs.tsv'
print('outputfile:', outputname)

# segments = []
# increment = ((endposition - startposition) // numthreads) + 1
# for floor in range(startposition, endposition, increment):
#     ceiling = floor + increment
#     if ceiling > endposition:
#         ceiling = endposition
#     segments.append((floor, ceiling))

packages = []
for centerdate, spanmeta in spanstocalculate:
    spandata = dict()
    
    for paperId, row in spanmeta.iterrows():  # the index is paperId
        for i in range(1000):
            chunkid = paperId + '-' + str(i)
            if i > 990:
                print('danger: ', i)
            if chunkid in data:
                spandata[chunkid] = data[chunkid]
            else:
                break

    package = (centerdate, spanmeta, spandata, exclusions, 'cosine')
    packages.append(package)

del data, meta, exclusions

print('Beginning multiprocessing.')
pool = Pool(processes = len(spanstocalculate))

res = pool.map_async(cpw.calculate_a_year, packages)
res.wait()
resultlist = res.get()
pool.close()
pool.join()
print('Multiprocessing concluded.')

for result in resultlist:
    doc_precocities, centerdate, condition_package = result
    fractions2check, filter_states, positive_radii, aggregates2check = condition_package

    if not os.path.isfile(summaryfile):
        with open(summaryfile, mode = 'w', encoding = 'utf-8') as f:
            outlist = ['docid', 'date', 'num_chunks', 'fraction_compared', 'filtered', 'time_radius', 'chunks_used', 'precocity', 'novelty', 'transience']
            header = '\t'.join(outlist) + '\n'
            f.write(header)

    with open(summaryfile, mode = 'a', encoding = 'utf-8') as f:
        for docid, doc_prec in doc_precocities.items():
            num_chunks = doc_prec['num_chunks']
            for frac in fractions2check:
                for filtered in filter_states:
                    for radius in positive_radii:
                        for aggregate in aggregates2check:
                            prec, nov, trans = doc_prec[(frac, filtered, radius, aggregate)]
                            outline = '\t'.join([str(x) for x in [docid, centerdate, num_chunks, frac, filtered, radius, aggregate, prec, nov, trans]])
                            f.write(outline + '\n')

    print(centerdate, 'written.')



