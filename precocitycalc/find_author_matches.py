# This script finds chunk pairs that cannot be compared because they share authors.
# This will be a subset of the "excluded_chunks."

# USAGE:
# python3 find_author_overlaps.py metadata_spreadsheet.tsv


# Updated May 18, 2024

import sys, json, os
import pandas as pd
from ast import literal_eval
from collections import Counter


def get_metadata(filepath):
	'''
	Loads the metadata spreadsheet and applies literal_eval to the
	authors column.
	'''
	meta = pd.read_csv(filepath, sep = '\t')
	meta['authors'] = meta['authors'].apply(literal_eval)
	
	return meta


def get_lowercase_last_names(author_names):
	'''
	#EDIT: add lowercase last names unless they're 'anonymous' 
	or in a lexicon of 25000 common words
	'''

	global lexicon
	lastnames = []
	for name in author_names:
		name = name.replace('\xa0', ' ')
		if name != 'anonymous':
			lastnames.append(name.split()[-1].lower())

	return [x for x in lastnames if x not in lexicon] 

def get_exclusions_for_all_files(metadata_df):
	'''
	This will iterate through all the files that have Semantic Scholar IDs in metadata_spreadsheet.
	In each case it will call get_exclusions(), which will return a list of forbidden papers.
	'''
	all_exclusions = dict()

	misses = 0
	hits = 0     # for debugging

	ctr = 0

	for idx, row in metadata_df.iterrows():
		if not pd.isnull(row['paperId']):        # this is checking to see whether we found it in S2
			pub_year = int(row.year)
			authors = row.authors
			cited_Id = row.paperId

			exclusions = get_exclusions(pub_year, authors, metadata_df)

			all_exclusions[cited_Id] = exclusions    # we store exclusions in a dict where key is the cited file Id
													 # and value is a list of forbidden chunk
			ctr += 1
			if ctr % 100 == 1:
				print(ctr)
	print('------')
	print(misses, hits)
	return all_exclusions

def get_exclusions(pub_year, cited_authors, metadata_df):

	forward_window = metadata_df.loc[(metadata_df.year > pub_year) & (metadata_df.year <= pub_year + 20) & (~pd.isnull(metadata_df.paperId)), : ]

	# We consider only papers published after the year of the cited-article's publication and within the next twenty years.

	exclusions = []
	cited_authorset = set(cited_authors)

	for idx, row in forward_window.iterrows():

		S2_Id = row.paperId

		# We're not going to compare any articles that share authorship. All chunks in such an article
		# are definitionally forbidden and all get added to the list of exclusions for this
		# cited_article.
		citing_authorset = set(row.authors)    

		if len(cited_authorset.intersection(citing_authorset)) > 0:  # this is a way of checking "if any are in" 
				exclusions.append(S2_Id)  

	return exclusions


# MAIN EXECUTION STARTS here

metadata_path = '/Users/tunder/Dropbox/python/novelty/metadata/litstudies/LitMetadataWithS2.tsv'

metadata = get_metadata(metadata_path)

all_exclusions = get_exclusions_for_all_files(metadata)   # does the actual work

print(len(all_exclusions))
with open('pairs_sharing_authors.tsv', mode = 'w', encoding='utf-8') as f:
	f.write('excludand\texcluded\n')
	for excludand, excludedlist in all_exclusions.items():
		for excluded in excludedlist:
			f.write(excludand + '\t' + excluded + '\n')
