# Measure Article Perplexity

# This script measures the perplexity of a set of articles using a pre-trained
# language model.

# It should be used on one four-year period at a time. We specify the period
# by identifying a twelve-year period of which the four-year period is the
# central third.

# The floor and ceiling of the twelve-year period are specified as command-line
# arguments, and together give us the name of the model to be used. (It will have been
# trained on the first four and last four years of the twelve-year period.)

import random, math, sys, torch, os
import pandas as pd

from transformers import RobertaConfig, RobertaModel, RobertaTokenizer
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

def LoadPaper(paperId, rootfolder):
    '''
    This function loads the text chunks corresponding to a paperId
    and returns them as a Dataset object.

    Parameters:
    -----------
    paperId: str    
        The paperId of the paper whose text chunks are to be loaded
    rootfolder: str
        The root folder containing the text files
    '''
    
    filepath = rootfolder + '/' + paperId + '.txt'
    with open(filepath, 'r') as file:
        for line in file:
            fields = line.strip().split('\t')
            if len(fields) != 2:
                continue
            text = fields[1]
            paper_Ids.append(paperId)
            text_data.append(text)

    df = pd.DataFrame({'paper_Id': paper_Ids, 'text': text_data})
    return Dataset.from_pandas(df)

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

def group_texts(examples, chunk_size=256):
    """
    Concatenates the texts in the given examples dataset, and breaks them
    into chunks of a specified size.

    Args:
        examples (dataset-dict): A dictionary containing the texts to be grouped.
        chunk_size (int, optional): The size of each chunk. Defaults to 256.

    Returns:
        dataset-dict: A dataset-dictionary containing the grouped texts, 
        with an additional "labels" column.

    """
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result

def all_word_masking(features):
    """
    Applies whole word masking to the input features, while multiplying them
    so that all the words in an example are masked one at a time, and there is
    an example for each word in the original example.

    It receives a single dataset, and returns a DatasetDict, where each key
    in the dataset corresponds to an index in the original Dataset, and the
    value is a dataset containing the masked examples (one for each word).

    Args:
        features (Dataset): A Dataset containing the following keys:
            - "word_ids" (List[int]): The tokenized word IDs.
            - "input_ids" (List[int]): The input token IDs.
            - "labels" (List[int]): The label token IDs.

    Returns:
        DatasetDict: A feature dictionary where each key corresponds to an index
        in the original Dataset, and the value is a Dataset containing the masked
        examples (one for each word). Also,
            - The "labels" key is updated with new label token IDs, which are -100 except
                for the masked tokens.
            - The "word_ids" key is removed from each Dataset.

    Raises:
        None

    """
    dataset_dict = {}

    for i, feature in enumerate(features):
        input_ids = feature["input_ids"]
        word_ids = feature["word_ids"] if "word_ids" in feature else None
        
        if word_ids is None:
            print("Error: No word_ids in feature.")
            continue

        # Mapping from word index to token indices
        mapping = {}
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id not in mapping:
                    mapping[word_id] = []
                mapping[word_id].append(idx)

        masked_datasets = []

        # Create masked versions: one mask per word
        for word_index in mapping:
            new_input_ids = input_ids.copy()
            new_labels = [-100] * len(input_ids)  # initially set all labels to -100

            for token_index in mapping[word_index]:
                new_input_ids[token_index] = tokenizer.mask_token_id
                new_labels[token_index] = input_ids[token_index]  # Set the correct label

            masked_datasets.append({"input_ids": new_input_ids, "labels": new_labels})

        dataset_dict[i] = Dataset.from_dict(masked_datasets)

    return DatasetDict(dataset_dict)

# Function to calculate perplexity
def calculate_perplexity(log_probs, num_tokens):
    """
    Calculates perplexity from log probabilities and the number of tokens.

    Args:
        log_probs (float): Sum of log probabilities for masked tokens.
        num_tokens (int): Number of masked tokens.

    Returns:
        float: The calculated perplexity.
    """
    if num_tokens > 0:
        return torch.exp(-torch.tensor(log_probs) / num_tokens).item()
    else:
        return float('inf')  # return infinity if no tokens to avoid division by zero

# Function to calculate model perplexities
def calculate_perplexities_for_model(model, data_loader):
    '''
    Given a model and a data_loader, iterate across the data and
    calculate the perplexity of the model on the data.
    '''
    total_log_probs = 0.0
    total_masked_tokens = 0
    for batch in data_loader:
        batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits

        labels = batch['labels']
        mask = labels != -100

        masked_logits = logits[mask]
        masked_labels = labels[mask]

        log_probs = torch.nn.functional.log_softmax(masked_logits, dim=-1)
        selected_log_probs = log_probs.gather(dim=-1, index=masked_labels.unsqueeze(-1)).squeeze(-1)

        total_log_probs += selected_log_probs.sum().item()
        total_masked_tokens += mask.sum().item()

    return calculate_perplexity(total_log_probs, total_masked_tokens)

# MAIN CODE EXECUTION STARTS HERE

args = sys.argv
metadatapath = args[1]
modelfloor = int(args[2])
modelceiling = int(args[3])
rootfolder = args[4]

# The floor and ceiling are inclusive endpoints of a twelve-year
# period that has been modeled.
assert modelfloor + 11 == modelceiling

# Infer the inner four-year period
floor = modelfloor + 4
ceiling = modelceiling - 4
assert ceiling - floor == 3

model_dir1 = 'from' + str(modelfloor) + 'to' + str(modelceiling)

# Load the configuration from the config.json file
config1 = RobertaConfig.from_pretrained(model_dir1)

# Load the RoBERTa model
model1 = RobertaModel(config1)
print('Model 1 loaded.')

# Also load models 12 and 16 years in the future
model_dir2 = 'from' + str(modelfloor + 12) + 'to' + str(modelceiling + 12)
model_dir3 = 'from' + str(modelfloor + 16) + 'to' + str(modelceiling + 16)
config2 = RobertaConfig.from_pretrained(model_dir2)
config3 = RobertaConfig.from_pretrained(model_dir3)
model2 = RobertaModel(config2)
model3 = RobertaModel(config3)

print('Models 2 and 3 loaded.')

model1name = 'model' + str(modelfloor) + '-' + str(modelceiling)
model2name = 'model' + str(modelfloor + 12) + '-' + str(modelceiling + 12)
model3name = 'model' + str(modelfloor + 16) + '-' + str(modelceiling + 16)

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

metadata = pd.read_csv(metadatapath, sep ='\t')   

print('Metadata loaded.')

# Assuming the models are already loaded, put them in evaluation mode
model1.eval()
model2.eval()
model3.eval()

# We will be storing the perplexities of each paper in a dictionary;
# the key will be the paperId, and the value will be a list of perplexities
# for each chunk of text in the paper. 
m1_paper_perplexities = {}
m2_paper_perplexities = {}
m3_paper_perplexities = {}

with open('PerplexitiesFrom' + str(floor) + 'To' + str(ceiling) + '.tsv', 'w') as file:
    file.write('paperId\tyear\tmodel\tindex\tperplexity\n')

for year in range(floor, ceiling + 1):
    paperIds_in_year = metadata[metadata['year'] == year]['paperId']
    for paper in paperIds_in_year:
        paper_dataset = LoadPaper(paper, rootfolder)
        tokenized_dataset = paper_dataset.map(tokenize_function, batched=True)
        grouped_dataset = tokenized_dataset.map(group_texts, batched=True)

        masked_dataset_dict = all_word_masking(grouped_dataset)

        m1_perplexities = []
        m2_perplexities = []
        m3_perplexities = []

        for key, masked_examples in masked_dataset_dict.items():
            data_loader = DataLoader(masked_examples, batch_size=32, collate_fn=default_data_collator)

            # Calculate and store perplexities for each model
            m1_perplexities.append(calculate_perplexities_for_model(model1, data_loader))
            m2_perplexities.append(calculate_perplexities_for_model(model2, data_loader))
            m3_perplexities.append(calculate_perplexities_for_model(model3, data_loader))

        m1_paper_perplexities[paper] = m1_perplexities
        m2_paper_perplexities[paper] = m2_perplexities
        m3_paper_perplexities[paper] = m3_perplexities

    # Save the perplexities to a file

    with open('PerplexitiesFrom' + str(floor) + 'To' + str(ceiling) + '.tsv', 'a') as file:
        for paper in m1_paper_perplexities.keys():
            for i, perplexity in enumerate(m1_paper_perplexities[paper]):
                file.write(paper + '\t' + str(year) + '\t' + model1name + '\t' + str(i) + '\t' + str(perplexity) + '\n')
            for i, perplexity in enumerate(m2_paper_perplexities[paper]):
                file.write(paper + '\t' + str(year) + '\t' + model2name + '\t' + str(i) + '\t' + str(perplexity) + '\n')
            for i, perplexity in enumerate(m3_paper_perplexities[paper]):
                file.write(paper + '\t' + str(year) + '\t' + model3name + '\t' + str(i) + '\t' + str(perplexity) + '\n') 

    print('Perplexities for year ', year, ' calculated and saved to file.')

    # Reinitialize the perplexity dictionaries for m1, m2, and m3
    m1_paper_perplexities = {}
    m2_paper_perplexities = {}
    m3_paper_perplexities = {}

print('Done.')