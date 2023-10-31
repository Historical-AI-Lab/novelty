import torch.nn.functional as F
from torch import Tensor
import torch 
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import pandas as pd
import sys, json, math

print('First imports complete.')

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

# USAGE

# We iterate through all the articles in the JSTOR file, converting each
# to a list of embeddings and a list of chunks

# We write the embeddings to a single tsv keyed by chunk ID, which
# is the numeric part of the JSTOR id plus chunk index. I.e., the 
# embeddings for "http://www.jstor.org/stable/512209" would be recorded as
#
# J512209-0
# J512209-1
# etc
#
# We don't write all the chunks, but do for every hundredth file so
# we can inspect them and make sure everything is working as we expect.

startline = int(sys.argv[1])
if len(sys.argv) > 2:
	metapath = sys.argv[2]
else:
	metapath = '../metadata/litstudies/LitMetadataWithS2.tsv'

metadata = pd.read_csv(metapath, sep = '\t')
metadata = metadata.set_index('doi')

def turn_undivided_text_into_sentences(document_pages):
	'''
	This function accepts a document as a single string and turns it into sentences.

	It will be replaced by Sarah's code. In particular when we look at 19/20c books,
	Sarah's code will have to handle turning a list of pages (each a list of lines)
	into a list of complete sentences.

	But for the academic journal articles this is easier. Each is a single string.
	'''

	document_string = ' '.join(document_pages)
	
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

def embeddings_for_an_article(articlestring):
	'''
	This runs the whole process from input string to embeddings.
	'''
	global device

	sentences = turn_undivided_text_into_sentences(articlestring)
	embedding_df = turn_sentences_to_embedding_df(sentences)
	chunk_list = turn_embedding_df_to_chunks(embedding_df)

	del sentences
	del embedding_df

	# Create list-of-lists for batches
	chunk_size = 8
	num_batches = math.ceil(len(chunk_list) / chunk_size)
	batched_chunk_lists = [chunk_list[i * chunk_size: (i + 1) * chunk_size] for i in range(num_batches)]

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

notdone = 0
errors = 0
ctr = 0

startline = int(sys.argv[1])

outlines = []

increment = 5000

with open('tryagaindata.jsonl', encoding = 'utf-8') as f:

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

		if 'identifier' in json_obj:
			for idtype in json_obj['identifier']:
				if idtype['name'] == 'paperId':
					paperId = idtype['value']
					foundmatch = True

		if not foundmatch:
			errors += 1
			outlines.append('error')
			continue

		else:
			article_text = json_obj['fullText']
			if len(article_text) > 100:
				chunk_list, embeddings = embeddings_for_an_article(article_text)
			else:
				outlines.append('short text: ' + paperId + " | " + article_text)
				continue

		outlines.append(str(json_obj['wordCount']) + ' | ' + str(paperId) + ' | ' + str(len(chunk_list)))

		with open('embeddingsTry2' + str(startline) + '.tsv', mode = 'a', encoding = 'utf-8') as f2:
			for i, e in enumerate(embeddings):
				f2.write(paperId + '-' + str(i) + '\t' + '\t'.join([str(x) for x in e.tolist()]) + '\n')
	
		with open('chunks/' + paperId + '.txt', mode = 'w', encoding = 'utf-8') as f3:
			for i, c in enumerate(chunk_list):
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
