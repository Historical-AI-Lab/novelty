# This is just a placeholder right now for a top-level module
# that will find chunks to exclude from the novelty-transience
# calculation.

# USAGE:
# python3 find_excluded_chunks.py metadata_spreadsheet.tsv folder_containing_chunkfiles/ citation_jsonl_path output_path
#
# The first argument will be the metadata spreadsheet for e.g. literary studies or ecology
# articles. The second will be the path to a folder containing the actual chunks.
#
# Note the folder containing chunk files should end with a slash.

# Updated Tues Nov 7

import sys, json, os
import pandas as pd
from ast import literal_eval
import string
from nltk import ngrams
import nltk
from collections import Counter

#EDIT: Get a lexicon of 25000 common words

lexicon = set()

with open('MainDictionary.txt', encoding = 'utf-8') as f:
	for line in f:
		words = line.strip().split()
		lexicon.add(words[0].lower())
		if len(lexicon) >= 25000:
			break

def get_metadata(filepath):
	'''
	Loads the metadata spreadsheet and applies literal_eval to the
	authors column.
	'''
	meta = pd.read_csv(filepath, sep = '\t')
	meta['authors'] = meta['authors'].apply(literal_eval)
	
	return meta

def load_citation_jsons(filepath):
	'''
	Gets citations, in the form of a dictionary where keys are cited paperIds
	and values are a list of citing paperIds.
	'''

	citations_for = dict()
	with open(filepath, encoding ='utf-8') as f:
		for line in f:
			json_obj = json.loads(line)
			if 'paperId' in json_obj and 'citations' in json_obj:
				paperId = json_obj['paperId']
				citations = json_obj['citations']
				citations_for[paperId] = citations
	return citations_for

def get_chunks(folder_containing_chunkfiles, S2_Id):

	'''
	Returns a list of 2-tuples where the first item is chunk_Id (defined as S2_Id + '-' + integer_order)
	and the second item is the text.
	'''

	chunklist = []
	desired_file = folder_containing_chunkfiles + S2_Id + '.txt'

	if not os.path.isfile(desired_file):
		return chunklist      # if we can't find the file we return an empty list

	with open(folder_containing_chunkfiles + S2_Id + '.txt', mode = 'r', encoding = 'utf-8') as f:
		for line in f:
			parts = line.split('\t')
			chunk_Id = S2_Id + '-' + parts[0]
			text = parts[1]
			chunklist.append((chunk_Id, text))

	return chunklist

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

def get_exclusions_for_all_files(metadata_df, folder_path, citations_for, output_log):
	'''
	This will iterate through all the files that have Semantic Scholar IDs in metadata_spreadsheet.
	In each case it will call get_exclusions(), which will return a list of forbidden chunks.

	Chunk id is S2_article_Id + '-'  + an integer.

	The code below is kind of pseudocode right now
	'''
	all_exclusions = dict()

	misses = 0
	hits = 0     # for debugging

	for idx, row in metadata_df.iterrows():
		if not pd.isnull(row['paperId']):        # this is checking to see whether we found it in S2
			pub_year = int(row.year)
			authors = row.authors
			cited_Id = row.paperId

			if cited_Id in citations_for:
				articles_that_cite_it = citations_for[cited_Id]  # a list of S2 Ids that cite the article in question
			else:
				articles_that_cite_it = []

			cited_chunks = get_chunks(folder_path, cited_Id)

			if len(cited_chunks) < 1:
				misses += 1
				continue        # if we can't find the file or it's empty there are no exclusions
			else:
				hits += 1
				with open('logofidsprocessed.txt', mode = 'a', encoding = 'utf-8') as f1:
					f1.write(cited_Id)    # for debugging purposes

			chunks_as_stripped_lists, had_quotes = strip_punctuation_from_chunks(cited_chunks)

			# We're going to turn each chunk into two things: 1) A list of lowercase words that have punctuation stripped
			# 2) a set of words that had quotes attached to them. Part 2 doesn't matter really for the cited_chunks
			# but will for the citing_chunks

			cited3grams = make_3grams(chunks_as_stripped_lists)  # This is just a set of 3grams (which are represented as tuples)

			exclusions = get_exclusions(cited_Id, pub_year, authors, cited3grams, articles_that_cite_it, metadata_df, folder_path)

			all_exclusions[cited_Id] = exclusions    # we store exclusions in a dict where key is the cited file Id
													 # and value is a list of forbidden chunks

			#FOR DEBUGGING WE COULD STOP IT AT 10 files

			# if hits > 10:
			#	break

			# INSTEAD I'M WRITING NON-EMPTY RESULTS TO LOG
			if len(exclusions) > 0:
				with open(output_log, mode = 'a', encoding = 'utf-8') as f:
					for excluded_chunk in exclusions:
						f.write(cited_Id + '\t' + excluded_chunk + '\n')



	print('------')
	print(misses, hits)
	return all_exclusions

def get_exclusions(cited_Id, pub_year, cited_authors, cited3grams, articles_that_cite_it, metadata_df, folder_path):

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
			citing_chunks = get_chunks(folder_path, S2_Id) # see function above for data structure returned: list of 2-tuples
			for chunk_Id, chunktext in citing_chunks:
				exclusions.append(chunk_Id)  
				pass
			continue # no need to check text if there are authors in common
		
		else:
			# Articles that don't share authors AND don't cite the cited_article are definitionally fine. We don't check them.
			# No exclusions get added for such an article, and we proceed to the next one in the
			# forward_window.

			if S2_Id not in articles_that_cite_it:
				continue
			else:
				citing_chunks = get_chunks(folder_path, S2_Id) # see function above for data structure returned: list of 2-tuples

		# Notice that we only read citing_chunks from file if we have to.
		# It's going to be very common that S2_Id is not in articles-that-cite-it,
		# and in that case, if there is no author overlap, we can skip the time-consuming file read.

		if len(citing_chunks) < 1:
			continue  # There's nothing to exclude if the file is empty or not found

		chunks_as_stripped_lists, had_quotes = strip_punctuation_from_chunks(citing_chunks)

		# We're going to turn each chunk into two things: 1) A list of lowercase words that have punctuation stripped
		# 2) a set of words that had quotes attached to them. Part 2 doesn't matter really for the cited_chunks
		# but will for the citing_chunks

		citing3grams = make_3grams(chunks_as_stripped_lists)  # This is just a list of sets of 3grams (which are represented as tuples)

		forbidden_chunks = get_forbidden_combos(cited3grams, citing3grams, had_quotes, cited_authors)

		if len(forbidden_chunks) > 0:
			exclusions.extend(forbidden_chunks)

	return exclusions

	
def flatten_set_of_tuples(tuples):
	word_list = []
	for tup in tuples:
		word_list.extend(tup)
	return word_list        # yep I think this works

def theres_an_author_match(citing_set, lowercase_last_names):
	list_of_words_in_chunk = flatten_set_of_tuples(citing_set)    # you need to put citing_set in the parens like all_words_in_chunk = flatten_set_of_tuples(citing_set)
	set_of_words_in_chunk = set(list_of_words_in_chunk)
	set_of_names = set(lowercase_last_names)
	names_found = set_of_words_in_chunk.intersection(set_of_names)

	if len(names_found) > 0:
		return True
	else:
		return False

def strip_punctuation_from_chunks(chunklist):
	'''
	Accepts a list of 2-tuples where each tuple contains
	a chunkid and
	a text

	and returns a list of 2-tuples where the text is 
	a list of words sans punctuation 
	'''

	newlist = []
	hadquotes = []

	for anid, text in chunklist:
		newlist.append((anid, strip_punctuation_from_list(text.split())))
		hadquotes.append(find_quoted_words(text))

	return newlist, hadquotes

def strip_punctuation_from_list(words):
	# Create a translation table that maps each punctuation character to None
	translation_table = str.maketrans('', '', string.punctuation)
	# Use the translation table to strip punctuation from each word in the list
	stripped_words = [word.translate(translation_table) for word in words]
	return stripped_words

def find_quoted_words(text):
	had_quotes = []
	tokenizedSent = text.split()
	for word in tokenizedSent:
		if word.startswith('"') or word.endswith('"'):
			had_quotes.append(word.strip('"'))
		elif word.startswith("'") or word.endswith("'"):
			had_quotes.append(word.strip("'"))

	return strip_punctuation_from_list(had_quotes)

def did_any_have_quotes(all_words_in_overlap, had_quotes):
	''' Returns a Boolean: True if any of the words in the set of 3 grams had
	quotes attached.
	'''
	quoted_overlap_words = [i for i in all_words_in_overlap if i in had_quotes]
	return len(quoted_overlap_words) > 0

def make_3grams(listof2tuples):
	'''
	This accepts a list of 2-tuples where each chunk is represented by
	a chunk id
	and a list of words

	it returns a list of 2-tuples where each chunk is represented by
	a chunk id
	and a set of 3 grams
	'''

	n = 3

	newlist = []

	for anid, listofwords in listof2tuples:
		words = ngrams(listofwords, n)
		grams = set()
		for gram in words:
			grams.add(gram)

		# EDIT: We also try adding 3grams with hyphen fusion to address EOL
		# breaks. In most cases these will not be new grams. A few will be new.

		hyphenreplaced = ' '.join(listofwords).replace('- ', '')
		fusedwordlist = hyphenreplaced.split()
		words = ngrams(listofwords, n)
		for gram in words:
			grams.add(gram)

		newlist.append((anid, grams))

	return newlist


def get_forbidden_combos(cited3grams, citing3grams, had_quotes, cited_authors):
	'''
	This function receives 

	a) A list, of length N1 where N1 is the number of chunks in the cited file. Each chunk 
	is represented as a set of 3grams.
	
	b) A list, of length N2 where N2 is the number of chunks in the citing file. Each chunk 
	is represented as a set of 3grams.

	Those 3grams are represented as tuples

	c) A list of length N2, where each chunk is represented as a set of words (individual words) *that had quotes of any kind attached*
	Single or double, at the beginning or at the end.

	d) cited_authors, in the format from the metadata_spreadsheet
	e.g. ['R. Weiss', 'Arthur Barker', 'George Kitchin', 'G. Tillotson', 'B. E. C. Davis', 'C. J. Sisson',
	 'W. M. T. Dodds', 'W. J. Entwistle', 'L. W. Tancock', 'F. J. Tanquerey', 'H. J. Hunt', 'A. C. Dunstan']

	The function's mission is to return a list of indexes in (b) where the chunk contained *either* a last name from d
	or at least six words in sequence shared with any chunk in a (and one of those words had quotes attached).

	'''
	forbidden = []
	ctr = 0
	for idx, citing_set in citing3grams:

		quoted_in_this_chunk = had_quotes[ctr]
		ctr += 1    # This counter is a kludge to keep "had quotes" aligned with
					# the chunk index. You'd think we would just use idx for that
					# but that variable has the whole S2_Id part of the chunk Id.

		lowercase_last_names = get_lowercase_last_names(cited_authors)
		if theres_an_author_match(citing_set, lowercase_last_names):   # you can write a function to check this
			forbidden.append(idx)
			print('AN AUTHOR NAME WAS FOUND')
			continue      # if any author names match any tokens in any of the 3grams, this citing chunk is forbidden
					# and we can proceed to the next
					# otherwise, we need to look for quotes
		for idx2, cited_set in cited3grams:
			# Then the next thing is, we don't have to compare the 3grams individually.
			# The point of having a set is you can do this ...
			overlapping3grams = cited_set.intersection(citing_set)    # .intersection() finds all the matches
			if len(overlapping3grams) < 4:
				# print(idx, len(overlapping3grams))      # a shared 6-word sequence will create I think four shared 3-grams
				continue                        # if we don't have four, there cannot be a shared 6-word sequence
												# so go to the next cited_set
			all_words_in_overlap = flatten_set_of_tuples(overlapping3grams)
							 # I'll imagine you've written a function that turns
							 # a set of tuples into a list of all the words in the tuples
							 # e.g [("review", "of", "economic"), ("of", "economic", "studies")]
							 # ==> ["review", "of", "economic", "of", "economic", "studies"]
			enough_repeats = count_repeats(all_words_in_overlap)
						# this is a function that checks to see if at least two words appear 3 or more times
						# and at least four words appear 2 or more times; this must be true if there's a
						# shared sequence for reasons explained at the end of this document
						# The function should return True or False
			anything_had_quotes = did_any_have_quotes(all_words_in_overlap, quoted_in_this_chunk)  # left as exercise
			if enough_repeats and anything_had_quotes:
				forbidden.append(idx)
				break      # we don't need to compare it to any other chunks of the cited document
						   # because one match is enough to condemn it
	return forbidden


#Another way to look at count_repeats:

def count_repeats(all_words_in_overlap):
	''' 
	'''
	word_counts = Counter(all_words_in_overlap)
	count_first_condition = sum(count >= 3 for count in word_counts.values()) >= 2
	count_second_condition = sum(count >= 2 for count in word_counts.values()) >= 4
	return count_first_condition and count_second_condition     # nice technique, combining them implicitly with and

#Test:
overlapping_words = ['what','is','the','economic','review','doing','about','this']
quoted_words = ['what', 'us', 'everything', "Hubble's", "don't"]
did_any_have_quotes(overlapping_words, quoted_words)

# MAIN EXECUTION STARTS here

metadata_path = sys.argv[1]
chunk_folder = sys.argv[2]
citation_jsonl_path = sys.argv[3]
output_log = sys.argv[4]

metadata = get_metadata(metadata_path)
citations_for = load_citation_jsons(citation_jsonl_path)   # a dictionary of citations for each paperId

all_exclusions = get_exclusions_for_all_files(metadata, chunk_folder, citations_for, output_log)   # does the actual work

print(len(all_exclusions))
print(all_exclusions)
