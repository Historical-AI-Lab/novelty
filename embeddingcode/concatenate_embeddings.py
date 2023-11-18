import os, glob

embedfiles = glob.glob('*embeddings*tsv')

prevdocs = set()
thesedocs = set()
duplicates = 0

for filename in embedfiles:
	with open(filename, encoding = 'utf-8') as f:
		for line in f:
			fields = line.strip().split()
			chunkID = fields[0]
			docID = chunkID.split('-')[0]
			if docID in prevdocs:
				duplicates += 1
			else:
				thesedocs.add(docID)

	prevdocs = prevdocs.union(thesedocs)
	thesedocs = set()


print(duplicates, len(prevdocs))


