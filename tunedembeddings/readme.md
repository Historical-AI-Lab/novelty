tunedembeddings
===============

This folder contains scripts for fine-tuning embeddings using sentence transformers and Multiple Negatives Ranking Loss. At the moment, this is the best way we've found of generating embeddings that represent the similarities and differences across a long timeline as successfully as a topic model.

The steps of the process (assuming you already have metadata and chunked texts):

1. Select training pairs, using one of the scripts that begin with `select_training_pairs_`. I used `select_training_pairs_final.py` to create the lit-studies model with 10% paraphrase (which was at the time called "final" lol). It selects pairs in four different categories, including successive passages from the same article, and pairs where one element will be a GPT-generated paraphrase of the other. (Note that it doesn't actually generate the paraphrase; it's just flagging that a paraphrase should be generated.) On the other hand `select_training_pairs_no_paraphrase.py` dispenses with the paraphrase option.

2. If your training data includes categories of pairs that need paraphrase ("single-synthetic" or "synthetic-pair"), you will need to run a Jupyter notebook that calls the OpenAI API to generate paraphrase. We could also set this up locally, but GPT-3.5 is so cheap it's hardly worth the effort. The most updated script is `synthetic_data_generator.ipynb`

3. The output of the data generator should then be rsynced up to Delta for training. (Do not, obviously, push to github as it will be big.) Put it in `/projects/bbiq/novelty/tunedembeddings.`

4. Run `TuneEmbeddings.py`. This is done on Delta, by editing a script like `train_nopara.slurm` or `train_para20.slurm`. The source of input data and the output folder are the main parameters that will need editing in the script. See argparse definitions at the top of the script.

5. Run `Apply_Tuned_Model.py`. This is also done on Delta, by editing a slurm script that begins with 'apply'. The modelpath and outputpath are the parameters that will need editing.

6. Rsync the embedding files produced by `Apply_Tuned_Model.py` over to the campus cluster, where we can run precocity calculation using lowly CPU time that is not as scarce as Delta time.