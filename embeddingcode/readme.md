embeddingcode
=============

Contains both python scripts and slurm scripts used in the production of GTE embeddings.

```embed_sentences.py``` produces sentence-length GTE embeddings; we're testing those against the longer embeddings produced by ```chunk_and_embed.py```. 

```MakeLitAfter1960.py``` produced a filtered version of the literary studies .jsonl for this experiment.

```sampleslurm.sbatch``` is the most updated version of the slurm script.