# novelty

Code used in the paper "Locating the Leading Edge of Cultural Change."

citation: 

```Sarah Griebel, Becca Cohen, Lucian Li, Jaihyun Park, Jiayu Liu, Jay Park, Jana Perkins, and Ted Underwood, Locating the Leading Edge of Cultural Change, Computational Humanities Research 24, Aarhus Denmark 2024,``` [```https://ceur-ws.org/Vol-3834/paper70.pdf```](https://ceur-ws.org/Vol-3834/paper70.pdf)

The paper itself is available here as [paper.pdf](paper.pdf)

There are two ways you can approach this repository: one is that you're curious about a claim in the paper and want to backtrack into the code to see where it was produced. In that case you probably want to start with the folder [```/interpret```](https://github.com/IllinoisLiteraryLab/novelty/tree/main/interpret), which contains notebooks that the paper rests on most immediately.

The other approach is if you're interested in trying to reproduce the whole process. That's more complex and is outlined below.

## Our general workflow

The basic logic is that we get texts and metadata--from JSTOR in the case of our nonfiction corpora, or from the Chicago Corpus in the case of fiction. Then, in the case of the nonfiction corpora, we match the articles to Semantic Scholar records in order to determine numbers of citations. (See ```/semantic_scholar``` and ```/metadata```.)

We perform three kinds of modeling on the texts: topic modeling, tuned SentenceBERT embeddings, and continued pretraining of RoBERTa models that we can use to estimate perplexity. (See ```/tunedembeddings```, ```/topicmodel```, and ```/perplexity```.) 

In the case of the first two methods, we then need to calculate *precocity* by comparing each text to texts in the future or past (represented as topic distributions or as embeddings). (Historically the scripts for this step were in ```/precocitycalc```, but the currently-used versions are distributed across ```/tunedembeddings``` and ```/topicmodel```.) The perplexity calculation is simpler and doesn't require an extra step.

At this point we have data that can be interpreted by the notebooks in ```/interpret```.

### authorage

Scripts that try to match our literary-studies authors to VIAF records, and then evaluate our confidence in the match. We used these--plus some manual work--to create a subset of literary-studies authors for which we had high confidence in their year of birth.

### cleandandchunk

Scripts that convert text files into embedding-sized chunks

### fiction

This folder is an exception to the rule that we tried to organize files according to the kind of representation (topic model, embeddings, etc) rather than subject domain. Some of the work on fiction--especially having to do with embeddings--is contained here.

### metadata

contains metadata for all our corpora

### semantic_scholar

aligning JSTOR metadata with semantic scholar metadata about citations

### precocitycalc

*Deprecated: some functions that were located here are now distributed elsewhere.*

This once contained code for doing forward-and-back calculations, which is now distributed across ```/tunedembeddings``` and ```/topicmodel```. Code for text reuse calculation is still placed here.

### topicmodel

instructions for generating topic models and then using those models to calculate precocity

### tunedembeddings

contains scripts for fine-tuning embeddings using sentence transformers, and then using those embeddings to calculate preocity

