# This is just a placeholder right now for a top-level module
# that will find chunks to exclude from the novelty-transience
# calculation.

# USAGE:
# python find_excluded_chunks.py metadata_spreadsheet.tsv folder_containing_chunkfiles citation_jsonl_path
#
# The first argument will be the metadata spreadsheet for e.g. literary studies or ecology
# articles. The second will be the path to a folder containing the actual chunks.

# Updated Tues Nov 7

import sys, json
import pandas as pd
from ast import literal_eval

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

def get_exclusions_for_all_files(metadata_df, folder_path, citations_for):
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

			articles_that_cite_it = citations_for[cited_Id]  # a list of S2 Ids that cite the article in question

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
def flatten_set_of_tuples(tuples):
	word_list = []
    for tup in tuples:
    	word_list.extend(tup)
	return word_list        # yep I think this works
def theres_an_author_match(citing_set, lowercase_last_names):
  flatten_set_of_tuples()    # you need to put citing_set in the parens like all_words_in_chunk = flatten_set_of_tuples(citing_set)
  author_match = False          # you could set this initially to False, and set it to True if match is found
  #check if author name matches any tokens in any of 3 grams
  for item in list:      # ah — you can simplify the nested loops by just saying if name in all_words_in_chunk: where you currently say if item == name:
    for name in lowercase_last_names:
      #if item == 'Anon'
      if item == name:
        #author_match.append(item) # just say author_match = True?  we don't care how many matches there are, so break?
        author_match = True

    #question -- would it be better to use some sort of similarity score here for last names instead of an exact match?

    # I think exact match is okay since we're only checking last name. Fuzzy match becomes necessary when there are first names OR initials etc etc
#function to find words with quotation marks (single or double) attached to them. We'll need this to compare later with overlapping words from the 3grams.
#It's not perfect because it still picks up apostrophes from conjunctions...

def find_quoted_words(text):
    tokenizedSent = text.split()
    had_quotes = [i for i in tokenizedSent if "\"" in i]
    had_quotes += [i for i in tokenizedSent if "\'" in i]
    had_quotes = [word.strip('\"\'.') for word in had_quotes]
    return had_quotes

# Example
example_text = 'We agree with \'everything\' except Hubble\'s claim that "what we don\'t know cannot kill us."'
find_quoted_words(example_text)

def did_any_have_quotes(all_words_in_overlap, had_quotes):
    quoted_overlap_words = [i for i in all_words_in_overlap if i in had_quotes]
    return quoted_overlap_words

def get_forbidden_combos(cited3grams, citing3grams, had_quotes, cited_authors):
forbidden = []
for idx, citing_set in enumerate(citing3grams):
  if theres_an_author_match(citing_set, lowercase_last_names):   # you can write a function to check this
        forbidden.append(idx)
        continue      # if any author names match any tokens in any of the 3grams, this citing chunk is forbidden
                      # and we can proceed to the next
                      # otherwise, we need to look for quotes
    for cited_set in cited3grams:
            # Then the next thing is, we don't have to compare the 3grams individually.
      # The point of having a set is you can do this ...
      overlapping3grams = cited_set.intersection(citing_set)    # .intersection() finds all the matches
            if len(overlapping3grams) < 4:      # a shared 6-word sequence will create I think four shared 3-grams
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
            anything_had_quotes = did_any_have_quotes(all_words_in_overlap, had_quotes)  # left as exercise
            if enough_repeats and anything_had_quotes:
                forbidden.append(idx)
                break      # we don't need to compare it to any other chunks of the cited document
                           # because one match is enough to condemn it
#Another way to look at count_repeats:

def count_repeats(all_words_in_overlap):
    words = all_words_in_overlap.split()
    word_counts = Counter(words)
    count_first_condition = any(count >= 3 for count in word_counts.values())
    count_second_condition = sum(count >= 2 for count in word_counts.values()) >= 4
    return count_first_condition and count_second_condition     # nice technique, separating the conditions cdproj allows us to change criteria later if we need to

#Test:
overlapping_words = ['what','is','the','economic','review','doing','about','this']
quoted_words = ['what', 'us', 'everything', "Hubble's", "don't"]
did_any_have_quotes(overlapping_words, quoted_words)

# MAIN EXECUTION STARTS here

metadata_path = sys.argv[1]
chunk_folder = sys.argv[2]
citation_jsonl_path = sys.argv[3]

metadata = get_metadata(metadata_path)
citations_for = load_citation_jsons(citation_jsonl_path)   # a dictionary of citations for each paperId

get_exclusions_for_all_files(metadata, chunk_folder, citations_for)   # does the actual work
