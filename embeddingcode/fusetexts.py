# fusetexts.py

import os

import re
import string

import numpy as np

from collections import Counter

# Function to split text into words and strip punctuation
def split_text_into_words(text):
	# Remove punctuation except for internal apostrophes and hyphens
	# by replacing them with spaces
	text = re.sub(r'(?<!\w)[^\s\w\'-]+|[^\s\w\'-]+(?!\w)', ' ', text)
	# Split text on whitespace
	words = text.split()
	
	return [w.lower() for w in words] # lowercase everything

def fuse_texts(tuples_list):
	fused_texts = []  # List to hold the result
	current_labels = []  # Labels for the current fused text
	current_text_words = []  # Words in the current fused text
	word_count = 0  # Word count for the current fused text

	for label, text in tuples_list:
		text_word_count = len(text.split())  # Count words in current text
		# Check if adding the current text would exceed the soft limit (512 words)
		if word_count + text_word_count > 512:
			# If the last text would be an orphan and adding it won't exceed 1100 words, add it
			if len(tuples_list) - tuples_list.index((label, text)) == 1 and word_count + text_word_count <= 1100:
				pass  # The text will be added below; this just prevents resetting
			# Else, if the current fused text is already over 512 words or adding the current text
			# would make it too long (over 1200 words), save the current fused text first.
			elif word_count >= 512 or word_count + text_word_count > 1100:
				fused_texts.append((current_labels, current_text_words))
				current_labels = []  # Reset the labels list
				current_text_words = []  # Reset the words list
				word_count = 0  # Reset the word count
			
		# Add the current text's words and label
		current_labels.append(label)
		current_text_words.extend(text.split())
		word_count += text_word_count

	# Add the last fused text if there's any text left to add
	if current_text_words:
		fused_texts.append((current_labels, current_text_words))

	return fused_texts

stoplist = {'the', 'a', 'an', 'of', 'and', 'in', 'to', 'is', 'that', 'for', 'by', 'with', 'on', 'not', 'which', 'this', 
'be', 'from', 'but', 'was', 'are', 'or', 'at', 'were'}


lexicon = set()
with open('lexicon.txt', encoding = 'utf-8') as f:
	for line in f:
		word, count = line.strip().split('\t')
		if word not in stoplist:
			lexicon.add(word)
		if len(lexicon) >= 50000:
			break

# List all files in 'chunks' directory
txt_files = [f for f in os.listdir('chunks') if f.endswith('.txt')]

sizedist = []
ctr = 0

# Iterate through the listed .txt files
for file in txt_files:
	# Construct full file path
	file_path = os.path.join('chunks', file)
	ssid = file.split('.')[0]
	ctr += 1
	if ctr % 100 == 1:
		print(ctr)
	
	# Process each file
	with open(file_path, 'r') as f:
		# Now you can work with the file's content
		contents = f.readlines()

		tuples_list = []

		for line in contents:
			lineparts = line.split('\t')
			chunknum = lineparts[0]
			text = lineparts[1]
			if len(lineparts) > 2:
				print('tab in text')
				text = ' '.join(lineparts[1:])
			tuples_list.append((chunknum, text))

		fused_texts = fuse_texts(tuples_list)

		with open ('litstudiesforLDA.txt', mode = 'a', encoding = 'utf=8') as f2:
			for labels, words in fused_texts:
				outtext = ' '.join([x for x in words if x in lexicon])
				chunkID = ssid + '-' + '.'.join(labels)
				outline = chunkID + '\t' + 'l1' + '\t' + outtext + '\n'
				f.write(outline)
				sizedist.append(len(outtext))

sizedist = np.array(sizedist)
for size in range(100, 600, 1200):
	count = sum((sizedist < size) & (sizedist > size - 100))
	print("Less than ", size, " : ", count)


