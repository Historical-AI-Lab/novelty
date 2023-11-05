# chunk size

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
	
	return words

# List all files in 'chunks' directory
txt_files = [f for f in os.listdir('chunks') if f.endswith('.txt')]

sizedist = []
lexicon = Counter()

# Iterate through the listed .txt files
for file in txt_files:
	# Construct full file path
	file_path = os.path.join('chunks', file)

	ctr += 1
	
	# Process each file
	with open(file_path, 'r') as f:
		# Now you can work with the file's content
		contents = f.readlines()
		
		for line in contents:
			text = line.split('\t')[1]
			words = split_text_into_words(text)
			wordcount = len(words)
			for w in words:
				lexicon[w] += 1

	if ctr % 500 == 2:
		print(ctr)

sizedist = np.array(sizedist)

for size in range(100, 600, 100):
	count = sum((sizedist < size) & (sizedist > size - 100))
	print("Less than ", size, " : ", count)

with open('lexicon.txt', mode = 'w', encoding = 'utf-8') as f:
	for word, count in lexicon.most_common(75000):
		f.write(word + '\t' + str(count) + '\n')




