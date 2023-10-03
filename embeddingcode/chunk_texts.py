import torch.nn.functional as F
from torch import Tensor
from torch import tensor
from transformers import AutoTokenizer, AutoModel
from scipy.spatial.distance import cosine
import pandas as pd
import json

print('First imports complete.')

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")

print('Tokenizer and model built.')

import nltk

nltk.download('punkt')

from nltk.tokenize import sent_tokenize

print('NLTK downloaded.')

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

	''' This is Becca's code, slightly adapted so we keep track of input ids, token type ids, and
	attention masks while aggregating sentences.

	As a result, we can return not only a list of chunks, but a batch_dict in the original format,
	with items now aggregated so they come as close as possible to 512 without going over. This
	may save us from having to run the tokenizer a second time.

	(Probably not a big deal with this corpus, but could save processing time later.)
	'''

	_512_counter = 0

	words_under_512 = []
	iis_under_512 = []
	ttis_under_512 = []
	ams_under_512 = []

	chunk_list = []
	batch_dict = {'input_ids': [], 'token_type_ids': [], 'attention_mask': []}

	for index, row in embedding_df.iterrows():
	    next_count = _512_counter + int(row['numtokens'])

	    if next_count < 512:
	        _512_counter = next_count
	        words_under_512.append(str(row['sentence']))
	        print(row['input_ids'])
	        iis_under_512.extend(row['input_ids'])
	        ttis_under_512.extend(row['token_type_ids'])
	        ams_under_512.extend(row['attention_mask'])

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

	tensor_batch_dict = {\
	'input_ids': torch.tensor(batch_dict['input_ids'], dtype=torch.long), 
	'token_type_ids': torch.tensor(batch_dict['token_type_ids'], dtype=torch.long),
    'attention_mask': torch.tensor(batch_dict['attention_mask'], dtype=torch.long)
    }

	return chunk_list, tensor_batch_dict

def embeddings_for_an_article(articlestring):
	'''
	This runs the whole process from input string to embeddings.
	'''
	sentences = turn_undivided_text_into_sentences(articlestring)
	embedding_df = turn_sentences_to_embedding_df(sentences)
	chunk_list, batch_dict = turn_embedding_df_to_chunks(embedding_df)

	outputs = model(**batch_dict)
	raw_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
	embeddings = F.normalize(raw_embeddings, p=2, dim=1)

	return chunk_list, embeddings

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

ctr = 0

with open('../LitStudiesJSTOR.jsonl', encoding = 'utf-8') as f:
	for line in f:
		json_obj = json.loads(line)

		article_text = json_obj['fullText'][0]
		chunk_list, embeddings = embeddings_for_an_article(article_text)

		articleID = json_obj['id'].replace('http://www.jstor.org/stable/', 'J')
		print(articleID, len(chunk_list))

		with open('embeddings.tsv', mode = 'a', encoding = 'utf-8') as f2:
			for i, e in enumerate(embeddings):
				f2.write(articleID + '-' + str(i) + '\t' + '\t'.join([str(x) for x in e]) + '\n')

		ctr += 1
		if ctr % 100 == 1:
			with open('chunks/' + articleID, mode = 'w', encoding = 'utf-8') as f3:
				for i, c in enumerate(chunk_list):
					f.write(str(i) + '\t' + c + '\n')






