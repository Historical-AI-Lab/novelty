# Model12Years.py is a script that trains a masked language model on two four-year shoulder periods
# of a twelve-year time segment. It breaks texts into 256-word chunks and uses whole-word masking to
# train and evaluate the model.

# Code written by Becca Cohen and Ted Underwood, drawing heavily on the
# HuggingFace blog post "Fine-Tuning a Masked Language Model."
# (https://huggingface.co/learn/nlp-course/en/chapter7/3)

# This version April 2024.

import random, math, sys, torch, os
import pandas as pd
from tqdm.auto import tqdm

from transformers import AutoModelForMaskedLM
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import TrainingArguments
import collections
import numpy as np
from transformers import default_data_collator
from transformers import AutoTokenizer
from transformers import Trainer, get_scheduler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from accelerate import Accelerator

print("Currently active directory:", os.getcwd())

def load_model_and_tokenizer(metadatapath = '../metadata/litstudies/LitMetadataWithS2.tsv'):
    """
    Loads a pre-trained model and tokenizer for masked language modeling.

    Args:
        metadatapath (str): The path to the metadata file (default: '../metadata/litstudies/LitMetadataWithS2.tsv')

    Returns:
        model (AutoModelForMaskedLM): The loaded pre-trained model for masked language modeling.
        tokenizer (AutoTokenizer): The loaded tokenizer for the model.
        metadata (pd.DataFrame): The loaded metadata as a pandas DataFrame.
    """
    model_checkpoint = "roberta-base"
    model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)

    metadata = pd.read_csv(metadatapath, sep ='\t')

    metadata['year'] = metadata['year'].astype(int)

    # Drop rows with missing paperId values
    metadata = metadata.dropna(subset=['paperId'])

    # Filter out paperIds with length less than 2
    metadata = metadata[metadata['paperId'].str.len() > 2]

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    return model, tokenizer, metadata

def LoadTimeSlice(floor, ceiling, metadata, rootfolder):
    '''
    This function loads data between floor and ceiling, and splits it into
    train and test sets.

    Parameters:
    -----------
    floor: int
    ceiling: int
    metadata: DataFrame
        A DataFrame containing metadata for the papers
    rootfolder: str
        The root folder containing the text files
    '''

    print('Selecting papers from the following years:')
    print('From', floor, 'to', ceiling, 'inclusive.')

    # Helper function to load data and split into train/test
    def load_period_data(floor, ceiling, metadata, rootfolder, split_ratio=0.2):
        paperIds = metadata[(metadata['year'] >= floor) & (metadata['year'] <= ceiling)]['paperId']
        text_data = []
        paper_Ids = []

        for paperId in paperIds:
            filepath = rootfolder + '/' + paperId + '.txt'
            if not os.path.exists(filepath):
                continue    
            with open(filepath, 'r') as file:
                for line in file:
                    fields = line.strip().split('\t')
                    if len(fields) != 2:
                        continue
                    text = fields[1]
                    paper_Ids.append(paperId)
                    text_data.append(text)

        df = pd.DataFrame({'paper_Id': paper_Ids, 'text': text_data})
        test_size = int(len(paperIds) * split_ratio)
        test_paperIds = paperIds.sample(n=test_size)
        test_set = df[df['paper_Id'].isin(test_paperIds)]
        training_set = df[~df['paper_Id'].isin(test_paperIds)]

        return Dataset.from_pandas(training_set), Dataset.from_pandas(test_set)

    # Load data for each period
    train, test = load_period_data(floor, ceiling, metadata, rootfolder)

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        'train': train,
        'test': test,
    })

    return dataset_dict

def tokenize_function(examples):
    """
    Tokenizes the input text using the tokenizer. Note that it truncates the text
    to a maximum length of 512 tokens.

    Args:
        examples (dataset-dict): A dataset-dictionary containing the input text.

    Returns:
        dataset-dict: A dataset-dictionary containing the tokenized text.

    """
    result = tokenizer(examples["text"], truncation=True, max_length=512)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def whole_word_masking_data_collator(features):
    """
    Applies whole word masking to the input features.

    Args:
        features (dataset-dict): A dataset dictionary containing the following keys:
            - "word_ids" (List[int]): The tokenized word IDs.
            - "input_ids" (List[int]): The input token IDs.
            - "labels" (List[int]): The label token IDs.

    Returns:
        dataset-dict: A list of modified feature dictionaries with the following changes:
            - The "labels" key is updated with new label token IDs, which are -100 except
                for the masked tokens.
            - The "word_ids" key is removed from each feature dictionary.

    Raises:
        None

    """
    global tokenizer
    wwm_probability = 0.15
    max_length = max(len(f['input_ids']) for f in features)
    
    for feature in features:
        if 'word_ids' in feature:
            word_ids = feature["word_ids"]
        else:
            print('error')
            break

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["input_ids"].copy()
        new_labels = [-100] * len(labels)
        
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            # Decide action for the whole word
            action = random.choices(['mask', 'random', 'preserve'], weights=[0.8, 0.1, 0.1])[0]
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]  # Set the label for learning

                if action == 'mask':
                    input_ids[idx] = tokenizer.mask_token_id
                elif action == 'random':
                    # Replace with a random token ID (make sure this is from your tokenizer's vocabulary)
                    input_ids[idx] = random.randint(0, tokenizer.vocab_size - 1)
                # If 'preserve', do nothing (leave the token as it is)

        padding_length = max_length - len(input_ids)
        feature['input_ids'] = input_ids + [tokenizer.pad_token_id] * padding_length
        feature['labels'] = new_labels + [-100] * padding_length  # Use -100 for padding token labels
        feature['attention_mask'] = feature['attention_mask'] + [0] * padding_length

    newfeatures = features.copy()
    for feature in newfeatures:
        feature.pop('word_ids')

    return default_data_collator(newfeatures)

## MAIN CODE EXECUTION STARTS HERE

# args = sys.argv

# metadatapath = args[1]
# floor = int(args[2])
# ceiling = int(args[3])
# rootfolder = args[4]

metadatapath = 'regressionsample.tsv'
rootfolder = './regressionchunks'

model, tokenizer, metadata = load_model_and_tokenizer(metadatapath)
print('Loaded model, tokenizer, and metadata')


dataset_dict = LoadTimeSlice(1900, 2020, metadata, rootfolder)
print('Successfully loaded dataset.')

tokenized_datasets = dataset_dict.map(
    tokenize_function, batched=True, remove_columns=["text", "paper_Id", '__index_level_0__']
)

# Print the labels and sizes of all datasets in lm_datasets
for dataset_label, dataset in tokenized_datasets.items():
    print(f"Dataset: {dataset_label}")
    print(f"Size: {len(dataset)}")
    print()

del dataset

# Create an output directory, unless it already exists.
output_dir = "./MLMtest"

if not os.path.exists(output_dir):
    print("Directory does not exist. Creating it.")
    os.makedirs(output_dir)
else:
    print("Directory exists.")

print()

def insert_random_mask(batch):
    '''
    This function inserts a random mask into the input data.
    It also changes column names, in a bit of a two-step process
    that will allow us to change the column names back later,
    after deleting the old columns.
    '''

    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = whole_word_masking_data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

# The number of test samples to select is the minimum of 2500 and the length of the smallest dataset.
# Note that it's important to select the same number of samples from the first four-year
# shoulder period and the second four-year shoulder period. Otherwise the model will be
# temporally biased.

test_dataset = tokenized_datasets['test'].shuffle(seed=42).select(range(8000))
train_dataset = tokenized_datasets['train'].shuffle(seed=42).select(range(50000))
print(train_dataset['attention_mask'][0])

eval_dataset = test_dataset.map(
    insert_random_mask,
    batched=True,
    remove_columns=test_dataset.column_names,
)

eval_dataset = eval_dataset.rename_columns(
    {
        "masked_input_ids": "input_ids",
        "masked_attention_mask": "attention_mask",
        "masked_labels": "labels"
    }
)

print('Training dataset length:', len(train_dataset))
print('Evaluation dataset length:', len(eval_dataset))

batch_size = 12
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=whole_word_masking_data_collator
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

optimizer = AdamW(model.parameters(), lr=4e-5)

# prepare model and data for acceleration

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 32
num_update_steps_per_epoch = len(train_dataloader)
num_training_steps = num_train_epochs * num_update_steps_per_epoch

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)

model.eval()
losses = []
for step, batch in enumerate(eval_dataloader):
    with torch.no_grad():
        outputs = model(**batch)

    loss = outputs.loss
    losses.append(accelerator.gather(loss.repeat(batch_size)))

losses = torch.cat(losses)
losses = losses[: len(eval_dataset)]
try:
    perplexity = math.exp(torch.mean(losses))
except OverflowError:
    perplexity = float("inf")

print(f">>> Starting condition: Perplexity: {perplexity}")

perplexitylist = [perplexity]
recent_perplexity = 0
older_perplexity = 0
trend = 100
convergedcount = 0

for epoch in range(num_train_epochs):
    # Training
    model.train()
    for batch in tqdm(train_dataloader, desc="Training"):
        outputs = model(**batch)
        loss = outputs.loss
        accelerator.backward(loss)

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    # Evaluation
    model.eval()
    losses = []
    for step, batch in enumerate(eval_dataloader):
        with torch.no_grad():
            outputs = model(**batch)

        loss = outputs.loss
        losses.append(accelerator.gather(loss.repeat(batch_size)))

    losses = torch.cat(losses)
    losses = losses[: len(eval_dataset)]
    try:
        perplexity = math.exp(torch.mean(losses))
    except OverflowError:
        perplexity = float("inf")

    perplexitylist.append(perplexity)

    # Our stopping condition is complex. We want to stop training if the model has converged,
    # but we assume that there will be some random up and down from epoch to epoch. So we
    # calculate a trend over six epochs, and if the trend is less than 0.1 for two consecutive
    # epochs, we assume convergence and stop training.
    
    if len(perplexitylist) > 5:
        recent_perplexity = sum(perplexitylist[-3:]) / 3
        older_perplexity = sum(perplexitylist[-6:-3]) / 3
        trend = round(older_perplexity - recent_perplexity, 3)
    
    if trend < 0.1:
        convergedcount += 1
    else:
        convergedcount = 0
    
    print(f">>> Epoch {epoch}: Perplexity: {perplexity}: Trend: {trend}")

    accelerator.wait_for_everyone()

    if (trend < 0.1 and convergedcount > 1) or epoch == num_train_epochs - 1:
        # Write the model to disk
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        break