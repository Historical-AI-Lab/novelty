workflow
========

This will be a guide to the workflow that takes an article dataset from original source (JSTOR metadata) through citation identification with Semantic Scholar, to production of embeddings and a topic model.

broad outline
-------------

1. Match the JSTOR metadata with Semantic Scholar using a script in `/datasources`. Sarah, since you've done this more recently, maybe you can help me figure out what's the up-to-date way of doing this?

2. Convert the JSTOR text into chunks of fewer than 512 tokens. This was originally fused with the embedding process, but we have rewritten the process to be cleaner. Now you start with a script in the `../cleanandchunk` folder -- which, as the label suggests, starts by cleaning the data (fusing words that have been broken across hy- phens at a line break). Then it divides into sentences and groups sentences so they constitute chunks of fewer than 512 tokens. It puts these in a folder, with each article's chunks named after a paperId from Semantic Scholar. Articles that lack paperIds are not processed.

3. Convert the chunks into embeddings using thenlper/gte-base embedding model. Most updated script is `embed_chunks_new.py`. The script takes as input a folder that contains .txt files labeled as paperID, each of which is a paper with each chunk on a new line along with its chunkID (separated by '\t'). It outputs a tsv file with column headers paperID, chunkID, and a list of embeddings for each chunk.

4. To topic model the corpus, see the instructions in the `./topicmodel` folder. It's a two-stage process because we want a flattish distribution across time, but then have to return and use a "topic inferencer" to generate topic distributions for things that were left out of the initial model (since those years poked up above the cap of the flat distribution).

5. *Text-reuse process described here.*
