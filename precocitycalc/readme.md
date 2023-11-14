precocity calculation
=====================

This folder will include the code we use to calculate novelty, transience, and precocity for both the topic model distributions, and the text embeddings.

Right now contains ```find_excluded_chunks.py```, a module we call to find chunks that can't be compared. We'll do this before running the novelty & transience calculations for a dataset.

The currently active version of this script is ```find_excluded_chunks_first_run.py``` which is being used for an initial pass on the literary studies corpus.