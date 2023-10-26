precocity calculation
=====================

This folder will include the code we use to calculate novelty, transience, and precocity for both the topic model distributions, and the text embeddings.

It will also contain ```find_excluded_chunks.py```, a module we call to find chunks that can't be compared. We'll do this either before running the novelty & transience calculations for a dataset, or perhaps *from* the script that does those calculations.