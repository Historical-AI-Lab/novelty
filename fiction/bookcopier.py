import SonicScrewdriver as utils

import argparse, os, shutil
import pandas as pd
from collections import Counter

# Command-line arguments include
# -i or --input: the list of docids to copy
# -o or --output: the output folder

parser = argparse.ArgumentParser(description="Copy files from one folder to another based on a list of docids")
parser.add_argument("-i", "--input", help="The list of docids to copy", required=True)
parser.add_argument("-o", "--output", help="The output folder", required=True)

args = vars(parser.parse_args())
outputfolder = args['output']
inputfile = args['input']

if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

meta = pd.read_csv(inputfile, sep = '\t')

def read_file(docid, date):
    if date < 1900:
        rootdir = '/projects/ichass/usesofscale/nonserials/'
    else:
        rootdir = '/projects/ichass/usesofscale/20c/'

    pairpath, postfix = utils.pairtreepath(docid, rootdir)

    filename = pairpath + postfix + '/' + postfix + ".norm.txt"
    if os.path.exists(filename):
        new_filename = os.path.join(outputfolder, docid + '.txt')
        shutil.copy(filename, new_filename)
        result = 'success'
    else:
        result = 'fail'
    
    return result

def main():
    results = []
    ctr = 0
    for idx, row in meta.iterrows():
        ctr += 1
        if ctr % 100 == 1:
            print(ctr)
        docid = row['docid']
        date = int(row['firstpub'])
        if date > 1925:
            continue
        result = read_file(docid, date)
        results.append(result)
    
    print(Counter(results))
    meta['fulltext'] = results
    meta.to_csv(os.path.join(outputfolder, 'books_added.tsv'), index = False, sep = '\t')

if __name__ == '__main__':
    main()

