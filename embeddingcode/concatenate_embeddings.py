import os, glob

embedfiles = glob.glob('*embeddings*tsv')

prevdocs = set()
thesedocs = set()
duplicates = 0

for filename in embedfiles:
	with open(filename, encoding = 'utf-8') a f:
		for line in f:
			fields = line.strip().split()
			chunkID = fields[0]
			docID = chunkID/split('-')[0]
			if doc in prevdocs:
				duplicates += 1
			else:
				thesedocs.add(doc)

	prevdocs = prevdocs.union(thesedocs)
	thesedocs = set()


print(duplicates, leb(prevdocs))


