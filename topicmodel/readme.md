topicmodel
==========

We won't push the raw data files to repo, so this folder will mostly contain the python scripts we use to produce raw data for modeling, and the SLURM scripts we used to direct Mallet to produce topic models.

initial topic modeling workflow
-------------------------------

```fusetexts.py``` is the script I used to fuse the individual article .txt files into one massive file for topic modeling. It combines chunks until they have *at least* 512 words, because many actual chunks had 200 or less, which is a bit too small for modeling. It also creates a lexicon, lowercases everything, and applies a very short list of stopwords (determiners, prepositions, conjunctions, and the verb to be).

```makelitnovelty.sbatch``` converted this text file into a .mallet data file

```litstudiesmodel2.sbatch``` actually ran the model for 2000 iterations, with 250 topics.

revised topic modeling workflow
-------------------------------

I realized that we needed a flatter distribution across time to avoid skewing the results, if we intend to draw any diachronic conclusions.

The approach I took to this depended on the fact that we had already used fusetexts.py to create '../embeddingcode/litstudiesforLDA.txt', which was the original text data used for topic modeling.

So I wrote two scripts: `ResampleTopicModelData.py` and `ExcludedFromFlat.py`.

The first of these, the resampler, selected a (relatively) flat distribution by counting the number of chunks per year in the original data and capping the new dataset at the number-per-year in the 40th-lowest year (40 from the bottom). This has the effect of producing a distribution that is flat from 1945 to 2015 or so, and tails off at either end, especially below 1945.

The new, flat dataset was in `ResampledLitStudiesForLDA.txt`. I needed to convert this into a `.mallet` file for topic modeling. I did that by running `makeflatmalletdata.slurm` (note that I run it in the mallet folder on the campus cluster -- `/projects/ischoolichass/ichass/usesofscale/mallet`), not in the novelty repo.

This produced `LitstudiesFlat.mallet`. I modeled that using `makeflatlitmodel.slurm` (again in the mallet directory).

This produced a bunch of data files, along with an inferencer that we could use to extend the model. This was necessary because our "flatter" distribution had sort of by definition left out a lot of texts that constituted the "bulge in the middle" of the old distribution.

To address this, we go back to `ExcludedFromFlat.py`, which basically constructs a text file containing everything in the original dataset that got left out of the resampled one. I called this `ExcludedFromFlat.txt`, and like all the .txt and .mallet files I'm describing it's on the campus cluster rather than in this repo. Too big.

Now we transform that into a .mallet file, using `make_lit_excluded_from_flat.slurm` (again run in the mallet directory), and apply the inferencer to it using `exclflatinferencer.slurm`.

The net result of all this is that I come away with two doctopics files to analyze: `excl_from_flat_doctopic_best.txt` and `flat250doctopics.txt`.

**Note:** In rerunning this workflow, it is probably better to simplify the whole thing and rewrite `fusetexts.py` so that *it* does the resampling, outputting both a flat-distribution text file and a text file that contains the topic-model-sized chunks excluded from the flat distribution. After this rewrite, `ResampleTopicModelData.py` and `ExcludedFromFlat.py` would not be necessary. You'd still need the slurm scripts though.
