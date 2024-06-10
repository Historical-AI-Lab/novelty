cleanandchunk
=============

This folder contains a utility for turning

1. A jsonl file from JSTOR + plus
2. A metadata file enriched with paperIds from Semantic Scholar

into a folder of files, where each file is named `paperId + .txt` and contains a sequence of chunks, each of which is made up of sentences that add up to fewer than 512 tokens

The format of the chunk file is that each line corresponds to a chunk, and has this format:

chunkidx (an integer) + \t + chunktext + '\n'

```USAGE:

python3 clean_and_chunk.py -s 3000 -d jsonl_file -m metadata_file -o outpath -l logfile

Where the command line arguments are, in order

-s               The line to *start* on in the jsonl_file

-d               Path to the .jsonl file containing full text *data* from JSTOR

-m               metadata_file -- Path to a *metadata* file that has been enriched by
                  our semantic scholar scripts. In particular, it should
                  have a paperId column from Semantic Scholar. paperIds will of course only exist
                  for articles we could find in Semantic Scholar.

-o               Path to the *output folder* where chunkfiles will be written

-l               Path to the *logfile* where we will write errors and other messages```