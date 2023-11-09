# This is just a placeholder right now for a top-level module
# that will find chunks to exclude from the novelty-transience
# calculation.

# USAGE:
# python find_excluded_chunks.py metadata_spreadsheet.tsv folder_containing_chunkfiles
#
# The first argument will be the metadata spreadsheet for e.g. literary studies or ecology
# articles. The second will be the path to a folder containing the actual chunks.

# Updated Tues Nov 7

import sys
import pandas as pd
from ast import literal_eval

def get_metadata(filepath):
	meta = pd.read_csv(filepath, sep = '\t')
	meta['authors'] = meta['authors'].apply(literal_eval)
	
	return meta

def get_chunks(folder_containing_chunkfiles, S2_Id):

	'''
	Returns a list of 2-tuples where the first item is chunk_Id (defined as S2_Id + '-' + integer_order)
	and the second item is the text.
	'''

	chunklist = []
	with open(folder_containing_chunkfiles + S2_Id + '.txt', mode = 'r', encoding = 'utf-8') as f:
		for line in f:
			parts = line.split('\t')
			chunk_Id = S2_Id + '-' + parts[0]
			text = parts[1]
			chunklist.append((chunk_Id, text))

	return chunklist

def lowercase_last_names(author_names):
    lastnames = []
    for name in author_names:
        if name != 'anonymous':
            lastnames.append(name.split()[-1].lower())
    return lastnames

def get_exclusions_for_all_files(metadata_df, folder_path):
	'''
	This will iterate through all the files that have Semantic Scholar IDs in metadata_spreadsheet.
	In each case it will call get_exclusions(), which will return a list of forbidden chunks.

	Chunk id is S2_article_Id + '-'  + an integer.

	The code below is kind of pseudocode right now
	'''
	all_exclusions = dict()

	for idx, row in metadata_df.iterrows():
		if not pd.isnull(row['paperId']):        # this is checking to see whether we found it in S2
			pub_year = int(row.year)
			authors = row.authors
			cited_Id = row.paperId

			cited_chunks = get_chunks(cited_Id)
			chunks_as_stripped_lists, had_quotes = strip_punctuation(chunks)

			# We're going to turn each chunk into two things: 1) A list of lowercase words that have punctuation stripped
			# 2) a set of words that had quotes attached to them. Part 2 doesn't matter really for the cited_chunks
			# but will for the citing_chunks

			cited3grams = make_3grams(chunks_as_stripped_lists)  # This is just a set of 3grams (which are represented as tuples)

			articles_that_cite_it = get_citing_Ids(for_this_article)  # a list of S2 Ids that cite the article in question

			exclusions = get_exclusions(cited_Id, pub_year, authors, cited3grams, articles_that_cite_it, metadata_df, folder_path):

			all_exclusions[cited_Id] = exclusions    # we store exclusions in a dict where key is the cited file Id
													 # and value is a list of forbidden chunks

	return all_exclusions

def get_exclusions(cited_Id, pub_year, cited_authors, cited3grams, articles_that_cite_it, metadata_df, folder_path):

	forward_window = metadata_df.loc[(metadata_df.year > pub_year) & (metadata_df.year <= pub_year + 20) & (~pd.isnull(metadata_df.paperId)), : ]

	# We consider only papers published after the year of the cited-article's publication and within the next twenty years.

	exclusions = []
	cited_authorset = set(cited_authors)

	for idx, row in forward_window.iterrows():

		S2_Id = row.paperId

		# Articles that don't cite the cited_article are definitionally fine. We don't check them.
		# No exclusions get added for such an article, and we proceed to the next one in the
		# forward_window.

		if S2_Id not in articles_that_cite_it:
			continue

		citing_chunks = get_chunks(S2_Id) # see function above for data structure returned: list of 2-tuples

		# We're not going to compare any articles that share authorship. All chunks in such an article
		# are definitionally forbidden and all get added to the list of exclusions for this
		# cited_article.
		citing_authorset = set(row.authors)
		if len(cited_authorset.intersection(citing_authorset)) > 1:  # this is a way of checking "if any are in" 
			for chunk_Id, chunktext in citing_chunks:
				exclusions.append(chunk_Id)  

			continue    

		chunks_as_stripped_lists, had_quotes = strip_punctuation(citing_chunks)

		# We're going to turn each chunk into two things: 1) A list of lowercase words that have punctuation stripped
		# 2) a set of words that had quotes attached to them. Part 2 doesn't matter really for the cited_chunks
		# but will for the citing_chunks

		citing3grams = make_3grams(chunks_as_stripped_lists)  # This is just a set of 3grams (which are represented as tuples)

		forbidden_chunks = get_forbidden_combos(cited3grams, citing3grams, had_quotes, cited_authors)

		exclusions.extend(forbidden_chunks)

	return exclusions


def get_forbidden_combos(cited3grams, citing3grams, had_quotes, cited_authors):

	'''
	This function receives 

	a) A list, of length N1 where N1 is the number of chunks in the cited file. Each chunk 
	is represented as a set of 3grams.
	
	b) A list, of length N2 where N2 is the number of chunks in the citing file. Each chunk 
	is representedas a set of 3grams.

	Those 3grams are represented as tuples

	c) A list of length N2, where each chunk is represented as a set of words (individual words) *that had quotes of any kind attached*
	Single or double, at the beginning or at the end.

	d) cited_authors, in the format from the metadata_spreadsheet
	e.g. ['R. Weiss', 'Arthur Barker', 'George Kitchin', 'G. Tillotson', 'B. E. C. Davis', 'C. J. Sisson',
	 'W. M. T. Dodds', 'W. J. Entwistle', 'L. W. Tancock', 'F. J. Tanquerey', 'H. J. Hunt', 'A. C. Dunstan']

	The function's mission is to return a list of indexes in (b) where the chunk contained *either* a last name from d
	or at least six words in sequence shared with any chunk in a (and one of those words had quotes attached).

	'''

# MAIN EXECUTION STARTS here

metadata_path = sys.argv[1]
chunk_folder = sys.argv[2]

metadata = get_metadata(metadata_path)

