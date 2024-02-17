# Calculate precocity

# This is based on previous calculate-kld scripts,
# but revised in 2023 to address a different
# research question as we evaluate different methods
# of measuring innovation.

# USAGE

# python calc_sentence_precocity.py metapath datapath excludepath startdate enddate function

# where

# startdate is inclusive
# enddate exclusive
# and function is either cosine or kld

import pandas as pd
import numpy as np
import random, sys, os
from multiprocessing import Pool
import calc_precocity_worker as cpw
from ast import literal_eval
from glob import glob

metapath = sys.argv[1]
datapath = sys.argv[2]
excludepath = sys.argv[3]
startdate = int(sys.argv[4])
enddate = int(sys.argv[5])
function_string = sys.argv[6]

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

# We have a challenge to handle. Our exclusions are defined to the smaller
# 512-token chunks, but the topic model combines those chunks to produce 
# larger ones of *at least* 512 words. So we need a map from the topic chunks
# to the exclusions

# Also notice data format is subtly different for the topic model. There's an
# initial unused field.

chunkmap = dict()

if function_string == 'cosine':
    files = glob(datapath + 'sent*.tsv')
    for afile in files:
        with open(afile, encoding = "utf-8") as f:
            for line in f:
                fields = line.strip().split('\t')
                chunkid = fields[0]
                vector = np.array([float(x) for x in fields[1:]], dtype = np.float64)
                data[chunkid] = vector
elif function_string == 'kld':
    with open(datapath, encoding = "utf-8") as f:
        for line in f:
            fields = line.strip().split('\t')
            chunkid = fields[1]
            vector = np.array([float(x) for x in fields[2:]], dtype = np.float64)
            data[chunkid] = vector
            chunkindexes = [x for x in chunkid.split('-')[1].split('.')]
            docid = chunkid.split('-')[0]
            for idx in chunkindexes:
                equivalent_chunk = docid + '-' + idx
                chunkmap[equivalent_chunk] = chunkid
else:
    print('Illegal function string.')
    sys.exit(0)

print(len(chunkmap))

# with open(excludepath, encoding = "utf-8") as f:
#     for line in f:
#         fields = line.strip().split('\t')
#         centerdoc = fields[0]
#         if centerdoc not in exclusions:
#             exclusions[centerdoc] = set()
#         for field in fields[1:]:
#             if function_string == 'cosine':
#                 exclusions[centerdoc].add(field)
#             elif function_string == 'kld':
#                 if field in chunkmap:
#                     exclusions[centerdoc].add(chunkmap[field])
#                 else:
#                     print(field)

totalvols = meta.shape[0]

spanstocalculate = []
for centerdate in range(startdate, enddate):
    df = meta.loc[(meta.year >= centerdate - 20) & (meta.year <= centerdate + 20) &
    (~pd.isnull(meta.paperId)), : ]
    df.set_index('paperId', inplace = True)
    spanstocalculate.append((centerdate, df))

outputname = 'precocity_' + function_string + '_' + str(startdate)
summaryfile = 'sentence_results/' + outputname + 's_docs.tsv'
print(outputname)

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
    spanexclude = dict()
    for paperId, row in spanmeta.iterrows():  # the index is paperId
        for i in range(1000):
            chunkid = paperId + '-' + str(i)
            if i > 990:
                print('danger: ', i)
            if chunkid in data:
                spandata[chunkid] = data[chunkid]
            elif chunkid in chunkmap:
                chunkid = chunkmap[chunkid]
                spandata[chunkid] = data[chunkid]
            else:
                break
        if row.year == centerdate and paperId in exclusions:
            spanexclude[paperId] = exclusions[paperId]


    package = (centerdate, spanmeta, spandata, spanexclude, function_string)
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



