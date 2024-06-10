workflow
========

This will be a guide to the workflow that takes an article dataset from original source (JSTOR metadata) through citation identification with Semantic Scholar, to production of embeddings and a topic model.

Right now it's mostly a placeholder.

broad outline
-------------

1. Match the JSTOR metadata with Semantic Scholar using a script in `/datasources`. Sarah, since you've done this more recently, maybe you can help me figure out what's the up-to-date way of doing this?
2. Convert the JSTOR text into chunks of fewer than 512 tokens. This is currently complicated. With literary studies files we did the chunking with `embeddingcode/chunk_and_embed.py`, which simultaneously produces GTE embeddings. But then at a later stage of the process we decided to clean up words broken across a hyphen. We did *that* with `perplexity/CleanChunkFiles.py`. Probably what we should really do at this point is rewrite the chunking script so that it does the cleaning at the same time, and separate out the making-GTE-embeddings part as a separate stage. We may not always need to do it, since GTE is not performing especially well.
3. The other thing that's an issue right now is that `chunk-and-embed` seems to be using the wrong signal to decide which articles to use. It's looking at the 'local doi' in the JSTOR metadata. But now that we have a fuller way of searching S2 for matches, we should probably just use all files that have a paperId?
