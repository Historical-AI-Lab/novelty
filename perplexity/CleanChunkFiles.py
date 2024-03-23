from glob import glob
import pandas as pd
import string

# CleanChunkFiles.py

# This script reads in the chunk files and cleans them by removing hyphens at the end of lines and 
# merging them with the next line if the merged word is in the dictionary. The cleaned chunks are 
# then written to new files.


chunkfiles = glob('/projects/ischoolichass/ichass/usesofscale/novelty/embeddingcode/chunks/*.txt')

dictionary = set()

with open('../precocitycalc/MainDictionary.txt', encoding='utf-8') as f:
    for line in f:
        word = line.split('\t')[0]
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

metadata = pd.read_csv('/projects/ischoolichass/ichass/usesofscale/novelty/metadata/litstudies/LitMetadataWithS2.tsv', sep='\t')
metadata['year'] = metadata['year'].astype(int)

newfolder = '/projects/ischoolichass/ichass/usesofscale/novelty/perplexity/cleanchunks/'
folderbefore29 = '/projects/ischoolichass/ichass/usesofscale/novelty/perplexity/cleanchunksbefore29/'

averagereductions = []

for path in chunkfiles:
    paperId = path.split('/')[-1].split('.')[0]
    cleanedfile = []
    tokencountdiffs = []
    
    with open(path, encoding='utf-8') as file:
        for line in file:
            line_parts = line.strip().split('\t')
            if len(line_parts) != 2:
                    print('Skipping line:', len(line))
                    continue
            chunk_number = int(line_parts[0])
            chunk_text = line_parts[1]
            # Process the chunk text here
            chunk_text = fix_broken_words(chunk_text, dictionary)
            cleaned_line = str(chunk_number) + '\t' + chunk_text
            cleanedfile.append(cleaned_line)
            original_length = len(line_parts[1].split())
            reduced_length = len(chunk_text.split())
            token_count_diff = (original_length - reduced_length) / original_length
            tokencountdiffs.append(token_count_diff)
    
    averagereduction = sum(tokencountdiffs) / len(tokencountdiffs)
    averagereductions.append(averagereduction)
    
    with open(newfolder + paperId + '.txt', 'w', encoding='utf-8') as outfile:
        outfile.write('\n'.join(cleanedfile))
    
    if paperId in metadata['paperId'].values:
        year = metadata.loc[metadata['paperId'] == paperId, 'year'].values[0]
        if year < 1929:
            with open(folderbefore29 + paperId + '.txt', 'w', encoding='utf-8') as outfile_before29:
                outfile_before29.write('\n'.join(cleanedfile))

print('Average token count reduction:', sum(averagereductions) / len(averagereductions))
