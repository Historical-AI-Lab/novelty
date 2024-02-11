# This version of the embedding script produces one-sentence
# chunks, unless the sentences are less than six words long.
# Those will be concatenated with an adjacent sentence.

# USAGE

#      python3 embed_sentences.py 3000 jsonl_file metadata_file

# Where the command line arguments are, in order
#
# 3000 -- The line to start on in the jsonl_file
# jsonl_file -- Path to the .jsonl file containing full text from JSTOR
# metadata_file -- Path to a metadata file that has been enriched by
#                  our semantic scholar scripts. In particular, it should
#                  have a doi column (from JSTOR) and a paperId column
#                  from Semantic Scholar. paperIds will of course only exist
#                  for articles we could find in Semantic Scholar.

# The script will produce a file called sent_embeddings_3000.tsv, which
# will contain the embeddings for the articles in the jsonl file, starting
# at line 3000. It also creates a directory called sentences — if it doesn't
# already exist. This directory will contain a file for each article, with
# the sentences in the article numbered and separated by tabs.

import torch.nn.functional as F
from torch import Tensor
import torch 
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import pandas as pd
import sys, json, math, os, string

print('First imports complete.')

# This section of code checks if the 'sentences' and 'sentence_embeddings' directories exist.
# If they don't exist, it creates them.

if not os.path.exists('sentences'):
	os.makedirs('sentences')
if not os.path.exists('sentence_embeddings'):
	os.makedirs('sentence_embeddings')

# Now we're going to parse the command-line arguments.

startline = int(sys.argv[1])

jsonlpath = sys.argv[2]

metapath = sys.argv[3]

metadata = pd.read_csv(metapath, sep = '\t')
metadata = metadata.set_index('doi')         # We're going to index metadata by doi
											 # It's important that this be the original doi
											 # provided by JSTOR, because we'll use it to
											 # align the jsons with our metadata file.

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")
model.eval()
model.to(device)

print('Tokenizer and model built.')

import nltk

nltk.download('punkt')

from nltk.tokenize import sent_tokenize

print('NLTK downloaded.')

dictionary = set()

with open('../precocitycalc/MainDictionary.txt', encoding = 'utf-8') as f:
	for line in f:
		word = f.split('\t')[0]
		dictionary.add(word)

def fix_broken_words(text, dictionary):
    # Split the text into words
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
                continue
            else:
                # If the merged word is not valid, just remove the hyphen
                word = word[:-1]
        # Add the current word (with or without modification) to the fixed words list
        fixed_words.append(word)
        i += 1
    
    # Join the fixed words back into a string
    return ' '.join(fixed_words)

def turn_undivided_text_into_sentences(document_string):
	'''
	This function accepts a document as a single string and turns it into sentences.
	It concatenates sentences if they're shorter than six words
	'''

	document_string = fix_broken_words(document_string)
	
	sentences = list(sent_tokenize(document_string))

	new_sentences = []
	last_sentence = []
	last_len = 0

	for s in sentences:
		wordlen = len(s.split())
		if wordlen < 6 and wordlen + last_len < 11:
			last_sentence.append(s)
			last_len += wordlen
		elif wordlen < 6 and wordlen + last_len < 320:
			last_sentence.append(s)
			sentence_to_add = ' '.join(last_sentence)
			new_sentences.append(sentence_to_add)
			last_sentence = []
			last_len = 0
		elif last_len < 320:
			sentence_to_add = ' '.join(last_sentence)
			new_sentences.append(sentence_to_add)
			last_sentence = [s]
			last_len = wordlen
		else:
			# Break the last_sentence into evenly-sized chunks
			last_sentence = ' '.join(last_sentence).split()  # Join sentences and split into words
			# Calculate the chunk size based on the length of last_sentence
			num_chunks = math.ceil(len(last_sentence) / 320)
			chunk_size = len(last_sentence) // num_chunks

			chunks = [last_sentence[i:i+chunk_size] for i in range(0, len(last_sentence), chunk_size)]
			for chunk in chunks:
				sentence_to_add = ' '.join(chunk)
				new_sentences.append(sentence_to_add)
			last_sentence = [s]
			last_len = wordlen

	return new_sentences 

def embeddings_for_an_article(articlestring):
	'''
	This runs the whole process from input string to embeddings.
	'''
	global device

	sentences = turn_undivided_text_into_sentences(articlestring)

	# Create list-of-lists for batches
	chunk_size = 8
	num_batches = math.ceil(len(sentences) / chunk_size)
	batched_chunk_lists = [sentences[i * chunk_size: (i + 1) * chunk_size] for i in range(num_batches)]

	# Initialize master list for all embeddings
	master_embeddings = []

	# Loop through each batch of chunk_list
	with torch.no_grad():   # save memory
		for batch in batched_chunk_lists:
			# Tokenize and move to device
			batch_dict = tokenizer(batch, max_length=512, padding=True, truncation=True, return_tensors='pt')
			batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

			# Generate embeddings
			outputs = model(**batch_dict)
			raw_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
			embeddings = F.normalize(raw_embeddings, p=2, dim=1)
			
			# Append to master list
			master_embeddings.append(embeddings)

			# Explicitly delete tensors to free memory
			del batch_dict
			del outputs
			del raw_embeddings

			# Clear GPU cache
			torch.cuda.empty_cache()

		master_embeddings = torch.cat(master_embeddings, dim=0)

	assert len(chunk_list) == len(master_embeddings)

	return chunk_list, master_embeddings

# MAIN
#
# Execution actually starts here.
#
# We iterate through all the articles in the JSTOR file, converting each
# to a list of embeddings and a list of chunks

# We write the embeddings to a single tsv keyed by chunk ID, which
# is the numeric part of the JSTOR id plus chunk index. I.e., the 
# embeddings for "http://www.jstor.org/stable/512209" would be recorded as
#
# J512209-0
# J512209-1
# etc

notdone = 0
errors = 0
ctr = 0

outlines = []

increment = 3000

outpath = 'sentence_embeddings/sent_embeddings_' + str(startline) + '.tsv'

# If the outfile for these 3000 lines already exists, we read it and
# make a note of the documents already embedded.

docswehave = set()

if os.path.exists(outpath):
	with open(outpath, mode = 'r', encoding = 'utf-8') as f:
		for line in f:
			chunkid = line.split('\t')[0]
			s2id = chunkid.split('-')[0]
			docswehave.add(s2id)

with open(jsonlpath, encoding = 'utf-8') as f:

	for line in f:
		if ctr >= startline + increment:
			break
		elif ctr >= startline:
			json_obj = json.loads(line)
			ctr += 1
			if ctr + 10 >= startline + increment:
				outstring = 'Errors: ' + str(errors) + '  Notdone: ' + str(notdone)
				outlines.append(outstring)
		else:
			ctr += 1
			continue

		foundmatch = False
		paperId = 'not a real Id'

		if 'identifier' in json_obj:
			for idtype in json_obj['identifier']:
				if idtype['name'] == 'local_doi':
					doi = idtype['value']
					if doi in metadata.index:
						proceedflag = metadata.at[doi, 'make_embeddings']
						paperId = metadata.at[doi, 'paperId']
						foundmatch = True

		if paperId in docswehave:
			continue

		if not foundmatch:
			errors += 1
			outlines.append('error')
			continue
		else:
			article_text = json_obj['fullText']
			if len(article_text) > 0 and len(article_text[0]) > 2:
				chunk_list, embeddings = embeddings_for_an_article(article_text)
			else:
				outlines.append('short text: ' + paperId)
				continue

		outlines.append(str(json_obj['wordCount']) + ' | ' + str(paperId) + ' | ' + str(len(chunk_list)))

		with open(outpath, mode = 'a', encoding = 'utf-8') as f2:
			for i, e in enumerate(embeddings):
				f2.write(paperId + '-' + str(i) + '\t' + '\t'.join([str(x) for x in e.tolist()]) + '\n')
	
		with open('sentences/' + paperId + '.txt', mode = 'w', encoding = 'utf-8') as f3:
			for i, c in enumerate(chunk_list):
				c = c.replace('\t', ' ').replace('\n', ' ')
				f3.write(str(i) + '\t' + c + '\n')

		if len(outlines) > 0:
			for outline in outlines:
				print(outline)
			outlines = []

if len(outlines) > 0:
	for outline in outlines:
		print(outline)
	outlines = []
print('Done. Execution complete.')
