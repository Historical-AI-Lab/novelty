import sys
import pandas as pd
import json

jsonlpath = '../LitStudiesJSTOR.jsonl'

metapath = '../metadata/litstudies/LitMetadataWithS2.tsv'

metadata = pd.read_csv(metapath, sep = '\t')
metadata = metadata.set_index('doi') 

outpath = 'FoundLitArticlesAfter1960.jsonl'

with open(jsonlpath, encoding='utf-8') as f:
    for line in f:
        json_obj = json.loads(line)
        if 'identifier' in json_obj:
            for idtype in json_obj['identifier']:
                if idtype['name'] == 'local_doi':
                    doi = idtype['value']
                    if doi in metadata.index:
                        proceedflag = metadata.at[doi, 'make_embeddings']
                        paperId = metadata.at[doi, 'paperId']
                        year = metadata.at[doi, 'year']
                        if proceedflag and year > 1960:
                            with open(outpath, 'a', encoding='utf-8') as out:
                                out.write(line)