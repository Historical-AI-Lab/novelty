# This is just a placeholder right now for a top-level module
# that will find chunks to exclude from the novelty-transience
# calculation.

# USAGE:
# python find_excluded_chunks.py metadata_spreadsheet.tsv folder_containing_chunkfiles
#
# The first argument will be the metadata spreadsheet for e.g. literary studies or ecology
# articles. The second will be the path to a folder containing the actual chunks.

def get_exclusions_for_all_files(metadata_df, folder_path):
	'''
	This will iterate through all the files that have Semantic Scholar IDs in metadata_spreadsheet.
	In each case it will call get_exclusions(), which will return a list of forbidden chunks.

	Chunk id is S2_article_Id + '-'  + an integer.

	The code below is kind of pseudocode right now
	'''
	all_exclusions = []

	for row in metadata_df:
		if row has S2_Id:
			pub_year = row.year
			authors = row.authors
			S2_Id = row.S2_Id

			cited_chunks = get_chunks(S2_Id)
			chunk_words, had_quotes = strip_punctuation(chunks)

			# We're going to turn each chunk into two things: 1) A list of lowercase words that have punctuation stripped
			# 2) a set of words that had quotes attached to them. Part 2 doesn't matter really for the cited_chunks
			# but will for the citing_chunks

			cited3grams = make_3grams(chunk_words)  # This is just a set of 3grams (which are represented as strings)

			exclusions = get_exclusions(S2_Id, pub_year, authors, cited3grams, metadata_df, folder_path):

			all_exclusions.append(exclusions)

def get_exclusions(S2_Id, pub_year, cited_authors, cited3grams, metadata_df, folder_path):

	forward_window = metadata_df.loc[year is in the next 20 years, and has S2_Id]

	exclusions = []

	for row in forward_window:

		citing_chunks = get_chunks(S2_Id)

		if any of row.authors are in authors: # We're not going to compare any articles that share authorship
			for chunk in citing_chunks:
				exclusions.append(this_chunk_ID)

			continue    

		chunk_words, had_quotes = strip_punctuation(citing_chunks)

		# We're going to turn each chunk into two things: 1) A list of lowercase words that have punctuation stripped
		# 2) a set of words that had quotes attached to them. Part 2 doesn't matter really for the cited_chunks
		# but will for the citing_chunks

		citing3grams = make_3grams(chunk_words)  # This is just a set of 3grams (which are represented as strings)

		forbidden_chunks = get_forbidden_combos(cited3grams, citing3grams, had_quotes, cited_authors)

		exclusions.extend(forbidden_chunks)


	return exclusions



