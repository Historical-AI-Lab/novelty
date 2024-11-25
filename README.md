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

In the case of the first two methods, we then need to calculate *precocity* by comparing each text to texts in the future or past (represented as topic distributions or as embeddings). (See ```/precocitycalc```.) The perplexity calculation is simpler and doesn't require an extra step.

At this point we have data that can be interpreted by the notebooks in ```/interpret```.

### cleandandchunk

scripts that convert text files into embedding-sized chunks

### semantic_scholar

contains scripts for aligning JSTOR metadata with semantic scholar metadata about citations.

### embeddingcode

This contains scripts for running GTE embeddings. Right now they are only adapted to run on data from JSTOR; getting them adapted for fiction is a next step.

### precocitycalc

*Deprecated: the functions that were located here are now distributed elsewhere.*

This contains code for doing forward-and-back calculations. It will apply to both the embedding and the topic model representations of the corpus. Code for text reuse calculation is also placed here, since excluding chunks that quote/paraphrase other documents is part of this workflow.

## topicmodel

instructions for generating topic models and then using those models to calculate precocity

## tunedembeddings

contains scripts for fine-tuning embeddings using sentence transformers, and then using those embeddings to calculate preocity

