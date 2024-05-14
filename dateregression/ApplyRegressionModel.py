# Apply Regression Model

# This script applies Roberta to predict date of publication.
# It also extracts embeddings from the model and saves them to a file.

# This version May 2024.

import random, math, sys, torch, os
import pandas as pd

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import Dataset, DatasetDict, concatenate_datasets
from transformers import TrainingArguments
import collections
import numpy as np
from scipy.stats import pearsonr
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

def load_model_and_tokenizer(input_dir, metadatapath):
    """
    Loads a pre-trained model and tokenizer for masked language modeling.

    Args:
        metadatapath (str): The path to the metadata file (default: '../metadata/litstudies/LitMetadataWithS2.tsv')
        input_dir: str
    Returns:
        model (AutoModelForMaskedLM): The loaded pre-trained model for masked language modeling.
        tokenizer (AutoTokenizer): The loaded tokenizer for the model.
        metadata (pd.DataFrame): The loaded metadata as a pandas DataFrame.
    """
    model = AutoModelForSequenceClassification.from_pretrained(input_dir, num_labels=1)

    metadata = pd.read_csv(metadatapath, sep ='\t')

    metadata['year'] = metadata['year'].astype(int)

    # Drop rows with missing paperId values
    metadata = metadata.dropna(subset=['paperId'])

    # Filter out paperIds with length less than 2
    metadata = metadata[metadata['paperId'].str.len() > 2]

    print(metadata.shape)

    tokenizer = AutoTokenizer.from_pretrained("roberta-base", truncation=True, max_length = 512)

    return model, tokenizer, metadata

def LoadDataset(metadata, rootfolder, floor, ceiling):
    '''
    This function loads data and returns it as a dataset including
    columns for year and paperId.

    Parameters:
    -----------
    metadata: DataFrame
        A DataFrame containing metadata for the papers
    rootfolder: str
        The root folder containing the text files
    floor: int
        The floor year for the dataset, inclusive
    ceiling: int
        The ceiling year for the dataset, inclusive
    '''
    text_data = []
    years = []
    papers = []

    for idx, row in metadata.iterrows():
        year = row['year']
        if year < floor or year > ceiling:
            continue
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

    data = Dataset.from_pandas(df)

    return data 

def preprocess_function(examples):
    label = examples["year"] 
    newexamples = tokenizer(examples["text"], truncation=True, max_length = 512)
    newexamples["label"] = label
    return newexamples

## MAIN CODE EXECUTION STARTS HERE

args = sys.argv

metadatapath = args[1]
floor = int(args[2])
ceiling = int(args[3])
rootfolder = args[4]
modelfolder = args[5]

# input_dir = "novelty/dateregression/AdaptedRegressor"

# metadatapath = 'novelty/dateregression/regressionsample.tsv'
# rootfolder = 'novelty/perplexity/regressionchunks'
# floor = 1917
# ceiling = 1918

print('Current directory:', os.getcwd())

outpath = './regoutput/regembeddings' + str(floor) + '-' + str(ceiling) + '.tsv'

model, tokenizer, metadata = load_model_and_tokenizer(modelfolder, metadatapath)
print('Loaded model, tokenizer, and metadata')

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

data = LoadDataset(metadata, rootfolder, floor, ceiling)

paperIds = data['paperId']
years = data['year']

tokenized_data = data.map(
    preprocess_function, batched=True, remove_columns=["paperId", "year", "text"]
)

tokenized_data.set_format("torch")

batch_size = 64
# Make sure to use the data collator you defined for padding
data_loader = DataLoader(tokenized_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

predicted_years = []
embeddings = []
ctr = 0

for batch in data_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']  # These correspond to your `years` normalized
    # You can now feed these batches to your model
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    predictions = outputs.logits.squeeze().tolist()
    predicted_years.extend(predictions)
    last_hidden_state = outputs.hidden_states[-1]  # This is the state input to the regression head
    mean_embeddings = torch.mean(last_hidden_state, dim=1)  # Average across tokens
    embeddings.append(mean_embeddings.cpu().numpy())
    ctr += 1
    if ctr % 4 == 1:
        print(ctr)

print(pearsonr(years, predicted_years))

# Convert embeddings list of arrays into a single numpy array
all_embeddings = np.concatenate(embeddings, axis=0)

# Create column names for the embeddings
embedding_columns = [f'embedding_{i}' for i in range(all_embeddings.shape[1])]

# Create DataFrame
outframe = pd.DataFrame({
    'paperId': paperIds,
    'year': years,
    'predicted_year': predicted_years
})

# Add embeddings to DataFrame
embeddings_df = pd.DataFrame(all_embeddings, columns=embedding_columns)
outframe = pd.concat([outframe, embeddings_df], axis=1)

# Save DataFrame to CSV
outframe.to_csv(outpath, sep='\t', index=False)