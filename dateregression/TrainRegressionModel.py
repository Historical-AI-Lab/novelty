# Train Regression Model

# This script trains Roberta to predict date of publication.

# This version May 2024.

import random, math, sys, torch, os
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import TrainingArguments
import collections
import numpy as np
from transformers import DataCollatorWithPadding
from transformers import Trainer, get_scheduler
from torch.utils.data import DataLoader
# from torch.optim import AdamW
# from accelerate import Accelerator

print("Currently active directory:", os.getcwd())

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    labels = labels.reshape(-1, 1)

    # Log some predictions to understand the output distribution
    if hasattr(compute_metrics, 'counter'):
        compute_metrics.counter += 1
    else:
        compute_metrics.counter = 1

    if compute_metrics.counter <= 50:  # Print first 5 batches
        print(f"Batch {compute_metrics.counter}:")
        print("Predictions:", logits.flatten()[:8])
        print("True Labels:", labels.flatten()[:8])
    
    mse = mean_squared_error(labels, logits)
    r2 = r2_score(labels, logits)
    
    return {"mse": mse, "r2": r2}

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
    model_checkpoint = "./MLMtest"
    model = AutoModelForSequenceClassification.from_pretrained(model_checkpoint, num_labels=1)

    metadata = pd.read_csv(metadatapath, sep ='\t')

    metadata['year'] = metadata['year'].astype(int)

    # Drop rows with missing paperId values
    metadata = metadata.dropna(subset=['paperId'])

    # Filter out paperIds with length less than 2
    metadata = metadata[metadata['paperId'].str.len() > 2]

    print(metadata.shape)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation=True, max_length = 512)

    return model, tokenizer, metadata

def LoadDatasets(metadata, rootfolder):
    '''
    This function loads data and separates it into training and test datasets.

    Parameters:
    -----------
    metadata: DataFrame
        A DataFrame containing metadata for the papers
    rootfolder: str
        The root folder containing the text files
    '''
    text_data = []
    years = []
    papers = []
    split_ratio = 0.15

    for idx, row in metadata.iterrows():
        year = row['year']
        paperId = row['paperId']
        filepath = rootfolder + '/' + paperId + '.txt'
        if not os.path.exists(filepath):
            continue
        with open(filepath, 'r') as file:
            for line in file:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    continue
                text = fields[1]
                papers.append(paperId)
                text_data.append(text)
                years.append(float(year - 1900) / 10)

    df = pd.DataFrame({'paperId': papers, 'text': text_data, 'year': years})
    test_size = int(len(metadata['paperId']) * split_ratio)
    print(test_size)

    test_paperIds = random.sample(metadata['paperId'].to_list(), test_size)
    test_set = df[df['paperId'].isin(test_paperIds)]
    training_set = df[~df['paperId'].isin(test_paperIds)]
    print(test_set.shape)
    print(training_set.shape)

    training_set = Dataset.from_pandas(training_set)
    test_set = Dataset.from_pandas(test_set)

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        'train': training_set,
        'test': test_set
    })

    return dataset_dict

def preprocess_function(examples):
    label = examples["year"] 
    newexamples = tokenizer(examples["text"], truncation=True, max_length = 512)
    newexamples["label"] = label
    return newexamples

## MAIN CODE EXECUTION STARTS HERE

# args = sys.argv

# metadatapath = args[1]
# floor = int(args[2])
# ceiling = int(args[3])
# rootfolder = args[4]

# metadatapath = 'novelty/metadata/litstudies/LitMetadataWithS2.tsv'
# rootfolder = 'novelty/perplexity/cleanchunksbefore29'

metadatapath = 'regressionsample.tsv'
rootfolder = './regressionchunks'

model, tokenizer, metadata = load_model_and_tokenizer(metadatapath)
print('Loaded model, tokenizer, and metadata')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

dataset_dict = LoadDatasets(metadata, rootfolder)

tokenized_train = dataset_dict['train'].map(
    preprocess_function, batched=True, remove_columns=["paperId", "year", "text", "__index_level_0__"]
)

tokenized_test = dataset_dict['test'].map(
    preprocess_function, batched=True, remove_columns=["paperId", "year", "text", "__index_level_0__"]
)   

tokenized_train.set_format("torch")
tokenized_test.set_format("torch")

tokenized_train = tokenized_train.shuffle(seed=42).select(range(50000))
tokenized_test = tokenized_test.shuffle(seed=42).select(range(8000))

# Print the labels and sizes of all datasets in lm_datasets
for dataset in [tokenized_train, tokenized_test]:
    print(f"Size: {len(dataset)}")
    # for key, feature in dataset.features.items():
    #     print(f"Feature: {key}")
    #     if isinstance(feature, str):
    #         print(f"Type: {feature}")
    #     else:
    #         print(f"Type: {feature._type}")
    print()

# Create an output directory, unless it already exists.
output_dir = "novelty/dateregression/AdaptedRegressor"

if not os.path.exists(output_dir):
    print("Directory does not exist. Creating it.")
    os.makedirs(output_dir)
else:
    print("Directory exists.")

print()

train_dataloader = DataLoader(tokenized_train, batch_size=32, shuffle=True)
test_dataloader = DataLoader(tokenized_test, batch_size=32, shuffle=True)

training_args = TrainingArguments(
    output_dir=output_dir,
    learning_rate=9e-5,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    num_train_epochs=16,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
)

# optimizer = torch.optim.Adam([
#     {'params': model.roberta.embeddings.parameters(), 'lr': 1e-5},  # Lower learning rate for embeddings
#     {'params': model.roberta.encoder.layer[:10].parameters(), 'lr': 5e-5},  # Lower learning rate for the first 10 layers
#     {'params': model.roberta.encoder.layer[10:].parameters(), 'lr': 1e-4},  # Higher learning rate for the remaining layers
#     {'params': model.classifier.parameters(), 'lr': 2e-4}  # Even higher learning rate for the classifier layer
# ], lr=1e-5)  # Default learning rate

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset = tokenized_train,
    eval_dataset = tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)
trainer.evaluate()
trainer.train()
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)