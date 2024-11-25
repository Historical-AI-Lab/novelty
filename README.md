# novelty

Code used in the paper "Locating the Leading Edge of Cultural Change."

citation: 

```Sarah Griebel, Becca Cohen, Lucian Li, Jaihyun Park, Jiayu Liu, Jay Park, Jana Perkins, and Ted Underwood, Locating the Leading Edge of Cultural Change, Computational Humanities Research 24, Aarhus Denmark 2024,``` [```https://ceur-ws.org/Vol-3834/paper70.pdf```](https://ceur-ws.org/Vol-3834/paper70.pdf)

The paper itself is available here as [paper.pdf](paper.pdf)

There are two ways you can approach this repository: one is that you're curious about a claim in the paper and want to backtrack into the code to see where it was produced. In that case you probably want to start with the folder ```/interpret```, which contains notebooks that the paper rests on most immediately.

The other approach is if you're interested in trying to reproduce the whole process. That's more complex and is outlined below.


## cleandandchunk

scripts that convert text files into embedding-sized chunks

## data sources

A lot of our current work is under here.

```/hathi_API``` contains a script for getting volumes page by page from Hathi.

```/semantic_scholar``` contains scripts for aligning JSTOR metadata with semantic scholar metadata about citations.

## embeddingcode

This contains scripts for running GTE embeddings. Right now they are only adapted to run on data from JSTOR; getting them adapted for fiction is a next step.

## precocitycalc

*Deprecated: the functions that were located here are now distributed elsewhere.*

This contains code for doing forward-and-back calculations. It will apply to both the embedding and the topic model representations of the corpus. Code for text reuse calculation is also placed here, since excluding chunks that quote/paraphrase other documents is part of this workflow.

## topicmodel

instructions for generating topic models and then using those models to calculate precocity

## tunedembeddings

contains scripts for fine-tuning embeddings using sentence transformers, and then using those embeddings to calculate preocity

