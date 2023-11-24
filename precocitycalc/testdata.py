topicpath = '../topicmodel/lits250doctopics.txt'
cosinepath = '../litstudies/all_lit_embeds.tsv'

topics = dict()

with open(topicpath, encoding = "utf-8") as f:
    for line in f:
        fields = line.strip().split('\t')
        chunkid = fields[0]
        docid = chunkid.split('-')[0]
        try:
        	chunkpart = chunkid.split('-')[1]
        except:
        	print(chunkid)
        	
        if docid not in topics:
        	topics[docid] = set()
        topics[docid].add(chunkpart)

cosines = dict()

with open(cosinepath, encoding = "utf-8") as f:
    for line in f:
        fields = line.strip().split('\t')
        chunkid = fields[1]
        chunkindexes = [x for x in chunkid.split('-')[1].split('.')]
        docid = chunkid.split('-')[0]
        if docid not in cosines:
        	cosines[docid] = set()
        for idx in chunkindexes:
            cosines[docid].add(idx)

for doc, chunks in cosines:
	if doc not in topics:
		print(doc, 'missing')
	else:
		missing = chunks - topics[doc]
		if len(missing) > 0:
			print(missing)