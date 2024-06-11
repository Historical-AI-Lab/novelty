# clean_and_chunk.py

# Adapted from chunk_and_clean.py, this script just divides
# JSTOR texts into chunks, while rejoining words divided
# across a line-ending hyphen. It does not in itself
# embed the chunks.

# Elements of this written by Cohen, Griebel, and Underwood.

# USAGE

#      python3 clean_and_chunk.py -s 3000 -d jsonl_file -m metadata_file -o outpath -l logfile

# Where the command line arguments are, in order
#
# -s               The line to *start* on in the jsonl_file
# -d               Path to the .jsonl file containing full text *data* from JSTOR
# -m               metadata_file -- Path to a *metadata* file that has been enriched by
#                  our semantic scholar scripts. In particular, it should
#                  have a paperId column from Semantic Scholar. paperIds will of course only exist
#                  for articles we could find in Semantic Scholar.
# -o               Path to the *output folder* where chunkfiles will be written
# -l               Path to the *logfile* where we will write errors and other messages

import pandas as pd
import json, os

import nltk

nltk.download('punkt')

from nltk.tokenize import sent_tokenize
print('NLTK downloaded.')

import torch
from transformers import AutoTokenizer
import string

import argparse

parser = argparse.ArgumentParser(description='Clean and Chunk')
parser.add_argument('-s', '--startline', type=int, help='The line to start on in the jsonl_file')
parser.add_argument('-d', '--data', type=str, help='Path to the .jsonl file containing full text data from JSTOR')
parser.add_argument('-m', '--metadata', type=str, help='Path to a metadata file that has been enriched by our semantic scholar scripts')
parser.add_argument('-o', '--outpath', type=str, help='Path to the output folder where chunkfiles will be written')
parser.add_argument('-l', '--logfile', type=str, help='Path to the logfile where we will write errors and other messages')

args = parser.parse_args()

startline = args.startline
jsonlpath = args.data
metapath = args.metadata
outpath = args.outpath     # A folder name that will be created if it does not yet exist
logfile = args.logfile

metadata = pd.read_csv(metapath, sep = '\t')
metadata = metadata.set_index('doi')         # We're going to index metadata by doi
											 # It's important that this be the original doi
											 # provided by JSTOR, because we'll use it to
											 # align the jsons with our metadata file.

dictionary = set()

with open('MainDictionary.txt', encoding='utf-8') as f:
	for line in f:
		word = line.split('\t')[0]
		dictionary.add(word)

fusecount = 0

print('Dictionary loaded and metadata read.')

def fix_broken_words(text, dictionary):
	# Split the text into words

	global fusecount

	words = text.split()
	fixed_words = []
	i = 0
	
	while i < len(words):
		word = words[i]
		# Check if the word ends with a hyphen and is not the last word in the list
		if word.endswith('-') and i + 1 < len(words):
			# Attempt to merge this word with the next one
			merged_word = word[:-1] + words[i + 1]
			# Check if the merged word is in the dictionary
			stripped_word = merged_word.strip(string.punctuation)
			if stripped_word.lower() in dictionary:
				# If the merged word is valid, add the merged word and skip the next one
				fixed_words.append(merged_word)
				i += 2
				fusecount += 1
				continue
			else:
				# If the merged word is not valid, just remove the hyphen
				word = word[:-1]
		# Add the current word (with or without modification) to the fixed words list
		fixed_words.append(word)
		i += 1
	
	# Join the fixed words back into a string
	return ' '.join(fixed_words)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")

def turn_undivided_text_into_sentences(document_string):
	'''
	This function accepts a document as a single string and turns it into sentences.

	It will be replaced by Sarah's code. In particular when we look at 19/20c books,
	Sarah's code will have to handle turning a list of pages (each a list of lines)
	into a list of complete sentences.

	But for the academic journal articles this is easier. Each is a single string.
	'''
	
	sentences = list(sent_tokenize(document_string))

	return sentences 

def turn_sentences_to_embedding_df(sentences):

	'''
	takes a list of sentences and turns it into a df with columns for the tokens, token types, 
	and attention_masks, as well as a column holding number of tokens for each sentence
	'''

	batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
	# note, we don't need to do this in a loop, because the tokenizer by default handles a list
	# of multiple input texts and outputs a batch_dict with multiple entries under each key,
	# corresponding to the texts it was given

	# let's confirm our understanding of this dictionary

	assert len(batch_dict['input_ids']) == len(batch_dict['attention_mask'])

	numsentences = len(batch_dict['input_ids'])

	numtokens = []

	for idx in range(numsentences):
		assert sum(batch_dict['attention_mask'][idx]) == sum(batch_dict['input_ids'][idx] > 0)

		# For each sentence, the total number of 1s in the attention_mask should be the same as the number of
		# nonzero input ids. This is how many tokens we have.

		numtokens.append(sum(batch_dict['attention_mask'][idx]))

	embedding_df = pd.DataFrame({'sentence': sentences, 'input_ids' : [x.tolist() for x in batch_dict['input_ids']],
		'token_type_ids': [x.tolist() for x in batch_dict['token_type_ids']], 
		'attention_mask': [x.tolist() for x in batch_dict['attention_mask']],
		'numtokens': numtokens})

	return embedding_df

def turn_embedding_df_to_chunks(embedding_df):

	''' This is Becca's code, with slightly different variable names.
	'''

	_512_counter = 0

	words_under_512 = []

	chunk_list = []
	
	for index, row in embedding_df.iterrows():
		next_count = _512_counter + int(row['numtokens'])

		if next_count < 512:
			_512_counter = next_count
			words_under_512.append(str(row['sentence']))
		

		elif next_count == 512:
			words_under_512.append(str(row['sentence']))
			chunk_list.append(' '.join(words_under_512))

			_512_counter = 0
			words_under_512 = []
		   

		else:  # next_count > 512
			# Do not append current sentence to words_under_512
			chunk_list.append(' '.join(words_under_512))
			words_under_512 = [str(row['sentence'])]  # Start new chunk with the "offending" sentence
			_512_counter = int(row['numtokens'])
	
	chunk_list.append(' '.join(words_under_512))

	return chunk_list

def chunks_for_an_article(document_pages, dictionary):
	'''
	This runs the whole process from input string to chunks.
	'''
	global device

	articlestring = ' '.join(document_pages)
	
	articlestring = fix_broken_words(articlestring, dictionary)

	sentences = turn_undivided_text_into_sentences(articlestring)
	embedding_df = turn_sentences_to_embedding_df(sentences)
	chunk_list = turn_embedding_df_to_chunks(embedding_df)

	del sentences
	del embedding_df

	return chunk_list

# MAIN
#
# Execution actually starts here.
#
# We iterate through all the articles in the JSTOR file, converting each
# to a list of chunks

# We write the embeddings to a single tsv keyed by chunk ID, which
# is the numeric part of the JSTOR id plus chunk index. I.e., the 
# embeddings for "http://www.jstor.org/stable/512209" would be recorded as
#
# J512209-0
# J512209-1
# etc

# MAXTOPROCESS = 1000 # This is a limit for testing. We will process at most this many articles.
MAXTOPROCESS = 1000000  # This is the production limit, which should be higher than the number of articles in the file.

notdone = 0
errors = 0
nulls = 0
ctr = 0

if not os.path.exists(outpath):
	os.makedirs(outpath)

files = os.listdir(outpath)
docswehave = set([filename.replace('.txt', '') for filename in files])

outlines = []

with open(jsonlpath, encoding = 'utf-8') as f:

	for line in f:
		if ctr >= startline:
			json_obj = json.loads(line)
			ctr += 1
		else:
			ctr += 1
			continue

		foundmatch = False
		paperId = 'not a real Id'

		if ctr % 100 == 1:
			print('Processing line: ', ctr)
		if ctr >= startline + MAXTOPROCESS:
			break

		if 'identifier' in json_obj:
			for idtype in json_obj['identifier']:
				if idtype['name'] == 'doi':
					doi = idtype['value']
					if doi in metadata.index:
						paperId = metadata.at[doi, 'paperId']
						if paperId is None or paperId == '' or pd.isna(paperId):
							paperId = 'null'
						foundmatch = True

		if paperId in docswehave:
			continue

		if not foundmatch:
			errors += 1
			outlines.append('No match: ' + str(ctr))
			continue
		elif paperId == 'null':
			outlines.append('Null paperId: ' + doi)
			nulls += 1
			continue
		else:
			document_pages = json_obj['fullText']
			if len(document_pages) > 0 and len(document_pages[0]) > 2:
				chunk_list = chunks_for_an_article(document_pages, dictionary)
			else:
				outlines.append('short text: ' + paperId)
				continue

		outlines.append(str(json_obj['wordCount']) + ' | ' + str(paperId) + ' | ' + str(len(chunk_list)))
	
		with open(outpath + '/' + paperId + '.txt', mode = 'w', encoding = 'utf-8') as f3:
			for i, c in enumerate(chunk_list):
				f3.write(str(i) + '\t' + c + '\n')

		if len(outlines) > 25:
			with open(logfile, mode = 'a', encoding = 'utf-8') as f:
				for outline in outlines:
					f.write(outline)
			outlines = []

if len(outlines) > 0:
	with open(logfile, mode = 'a', encoding = 'utf-8') as f:
		for outline in outlines:
			f.write(outline)
	outlines = []

print()
print('Errors:', errors)
print('Nulls:', nulls)
print('Words rejoined: ', fusecount)

print('Done. Execution complete.')
