# novelty

Research on different measures of novelty, precocity, and innovation. Begun in Fall 2023.

People involved include Rebecca Cohen, Sarah Griebel, Lucian Li, Jiayu Liu, Jay Park, Jana Perkins, and Ted Underwood.

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

