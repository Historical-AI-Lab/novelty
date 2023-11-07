topicmodel
==========

We won't push the raw data files to repo, so this folder will mostly contain the python scripts we use to produce raw data for modeling, and the SLURM scripts we used to direct Mallet to produce topic models.

```fusetexts.py``` is the script I used to fuse the individual article .txt files into one massive file for topic modeling. It combines chunks until they have *at least* 512 words, because many actual chunks had 200 or less, which is a bit too small for modeling. It also creates a lexicon, lowercases everything, and applies a very short list of stopwords (determiners, prepositions, conjunctions, and the verb to be).

```makelitnovelty.sbatch``` converts this text file into a .mallet data file

```litstudiesmodel2.sbatch``` actually runs the model for 2000 iterations, with 250 topics.