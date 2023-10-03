import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import pandas as pd

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")

import nltk
import csv

nltk.download('punkt')

from nltk.tokenize import sent_tokenize

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

	embedding_df = pd.DataFrame({'sentence': sentences, 'input_ids' : batch_dict['input_ids'],
		'token_type_ids': batch_dict['token_type_ids'], 'attention_mask': batch_dict['attention_mask'],
		'numtokens': numtokens})

	return embedding_df

def turn_embedding_df_to_chunks(embedding_df):

	''' This is Becca's code, slightly adapted so we keep track of input ids, token type ids, and
	attention masks while aggregating sentences.

	As a result, we can return not only a list of chunks, but a batch_dict in the original format,
	with items now aggregated so they come as close as possible to 512 without going over. This
	saves us from having to run the tokenizer a second time.

	(Probably not a big deal with this corpus, but maybe save processing time later.)
	'''

	_512_counter = 0

	words_under_512 = []
	iis_under_512 = []
	ttis_under_512 = []
	ams_under_512 = []

	chunk_list = []
	batch_dict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

	for index, row in df.iterrows():
	    next_count = _512_counter + int(row['numtokens'])

	    if next_count < 512:
	        _512_counter = next_count
	        words_under_512.append(str(row['sentence']))
	        iis_under_512.append(row['input_ids'].values)
	        ttis_under_512.append(row['token_type_ids'].values)
	        ams_under_512.append(row['attention_mask'].values)

	    elif next_count == 512:
	        words_under_512.append(str(row['sentence']))
	        iis_under_512.extend(row['input_ids'].values)
	        ttis_under_512.extend(row['token_type_ids'].values)
	        ams_under_512.extend(row['attention_mask'].values)

	        chunk_list.append(' '.join(words_under_512))
	        batch_dict['input_ids'].append(iis_under_512)
	        batch_dict['token_type_ids'].append(ttis_under_512)
	        batch_dict['attention_mask'].append(ams_under_512)

	        _512_counter = 0
	        words_under_512 = []
	        iis_under_512 = []
			ttis_under_512 = []
			ams_under_512 = []

	    else:  # next_count > 512
	        # Do not append current sentence to words_under_512
	        chunk_list.append(' '.join(words_under_512))
	        batch_dict['input_ids'].append(iis_under_512)
	        batch_dict['token_type_ids'].append(ttis_under_512)
	        batch_dict['attention_mask'].append(ams_under_512)

	        words_under_512 = [str(row['sentence'])]  # Start new chunk with the "offending" sentence
	        iis_under_512 = row['input_ids']
			ttis_under_512 = row['token_type_ids']
			ams_under_512 = row['attention_mask']
	        _512_counter = int(row['numtokens'])
	
	chunk_list.append(' '.join(words_under_512))
	batch_dict['input_ids'].append(iis_under_512)
	batch_dict['token_type_ids'].append(ttis_under_512)
	batch_dict['attention_mask'].append(ams_under_512)

	return chunk_list, batch_dict

