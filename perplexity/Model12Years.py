import random, math, sys, torch, os
import pandas as pd

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

# LoadTimeSlice
#
# This function accepts a floor, a ceiling, and a metadata dataframe.
# It selects paperIds between the floor and ceiling, and loads the text
# files corresponding to those paperIds. It transforms them first into
# a pandas dataframe and then into a HuggingFace DatasetDict, which
# it returns.

def OldLoadTimeSlice(floor, ceiling, metadata, rootfolder):
    '''
    This function creates a model that will be applied to the central four years
    of a twelve-year period. The model is trained only on the "shoulder" periods:
    the first four years and the last four.

    Parameters:
    -----------
    floor: int
        The first year of the twelve-year period
    ceiling: int
        The last year of the twelve-year period (inclusive)
    metadata: DataFrame
        A DataFrame containing metadata for the papers
    rootfolder: str
        The root folder containing the text files
    '''

    firstfloor = floor
    firstceiling = floor + 3
    secondfloor = floor + 8
    secondceiling = secondfloor + 3

    assert secondceiling == ceiling

    print('Selecting papers from the following years:')
    print('From', firstfloor, 'to', firstceiling, 'inclusive, and')
    print('From', secondfloor, 'to', secondceiling, 'inclusive.')

    selected_paperIds = metadata[((metadata['year'] >= firstfloor) & (metadata['year'] <= firstceiling)) | 
                                 ((metadata['year'] >= secondfloor) & (metadata['year'] <= secondceiling))]['paperId']

    text_data = []
    paper_Ids = []

    for paperId in selected_paperIds:
        # Load text file corresponding to paperId
        filepath = rootfolder + '/' + paperId + '.txt'
        with open(filepath, 'r') as file:
            for line in file:
                fields = line.strip().split('\t')
                if len(fields) != 2:
                    continue
                text = fields[1]
                paper_Ids.append(paperId)
                text_data.append(text)

    # Create a dataframe with paper_Ids and text_data
    data = {'paper_Id': paper_Ids, 'text': text_data}
    df = pd.DataFrame(data)

    # Randomly select 20% of the paperIds as test_set
    test_size = int(len(selected_paperIds) * 0.2)
    test_paperIds = selected_paperIds.sample(n=test_size)

    # Select corresponding rows of df as test_set dataframe
    test_set = df[df['paper_Id'].isin(test_paperIds)]

    # Select paperIds not in test_set as training_set
    training_set = df[~df['paper_Id'].isin(test_paperIds)]

    # Transform test_set and training_set into HuggingFace datasets
    test_dataset = Dataset.from_pandas(test_set)
    training_dataset = Dataset.from_pandas(training_set)

    # Combine test_dataset and training_dataset into a DatasetDict
    dataset_dict = DatasetDict({'train': training_dataset, 'test': test_dataset})

    return dataset_dict

def LoadTimeSlice(floor, ceiling, metadata, rootfolder):
    '''
    This function loads data from two shoulder periods of a twelve-year timeline
    and separates each shoulder period into training and test datasets.

    Parameters:
    -----------
    floor: int
        The first year of the twelve-year period
    ceiling: int
        The last year of the twelve-year period (inclusive)
    metadata: DataFrame
        A DataFrame containing metadata for the papers
    rootfolder: str
        The root folder containing the text files
    '''

    assert ceiling - floor == 11

    firstfloor = floor
    firstceiling = floor + 3
    secondfloor = ceiling - 3
    secondceiling = ceiling

    print('Selecting papers from the following years:')
    print('From', firstfloor, 'to', firstceiling, 'inclusive, and')
    print('From', secondfloor, 'to', secondceiling, 'inclusive.')

    # Helper function to load data and split into train/test
    def load_period_data(floor, ceiling, metadata, rootfolder, split_ratio=0.2):
        paperIds = metadata[(metadata['year'] >= floor) & (metadata['year'] <= ceiling)]['paperId']
        text_data = []
        paper_Ids = []

        for paperId in paperIds:
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
        test_size = int(len(paperIds) * split_ratio)
        test_paperIds = paperIds.sample(n=test_size)
        test_set = df[df['paper_Id'].isin(test_paperIds)]
        training_set = df[~df['paper_Id'].isin(test_paperIds)]

        return Dataset.from_pandas(training_set), Dataset.from_pandas(test_set)

    # Load data for each period
    first_train, first_test = load_period_data(firstfloor, firstceiling, metadata, rootfolder)
    second_train, second_test = load_period_data(secondfloor, secondceiling, metadata, rootfolder)

    # Combine into a DatasetDict
    dataset_dict = DatasetDict({
        'first_train': first_train,
        'first_test': first_test,
        'second_train': second_train,
        'second_test': second_test
    })

    return dataset_dict

def tokenize_function(examples):
    result = tokenizer(examples["text"], truncation=True, max_length=512)
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples, chunk_size=256):

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

def whole_word_masking_data_collator(features):
    wwm_probability = 0.15
    
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
        labels = feature["labels"]
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

        feature["labels"] = new_labels

    newfeatures = features.copy()
    for feature in newfeatures:
        feature.pop('word_ids')

    return default_data_collator(newfeatures)

## MAIN CODE EXECUTION STARTS HERE

args = sys.argv

metadatapath = args[1]
floor = int(args[2])
ceiling = int(args[3])
rootfolder = args[4]

# metadatapath = 'novelty/metadata/litstudies/LitMetadataWithS2.tsv'
# floor = 1910
# ceiling = 1921
# rootfolder = 'novelty/perplexity/cleanchunksbefore29'

model, tokenizer, metadata = load_model_and_tokenizer(metadatapath)
print('Loaded model, tokenizer, and metadata')


dataset_dict = LoadTimeSlice(floor, ceiling, metadata, rootfolder)
print('Successfully loaded dataset for', floor, 'to', ceiling)

tokenized_datasets = dataset_dict.map(
    tokenize_function, batched=True, remove_columns=["text", "paper_Id", '__index_level_0__']
)

## NOTE: the following parameter could be adjusted.
chunk_size_value = 256

lm_datasets = tokenized_datasets.map(group_texts, batched=True, fn_kwargs={'chunk_size': chunk_size_value})
print('Datasets grouped as 256-token chunks.')
print()

# Print the labels and sizes of all datasets in lm_datasets
for dataset_label, dataset in lm_datasets.items():
    print(f"Dataset: {dataset_label}")
    print(f"Size: {len(dataset)}")
    print()

del dataset, tokenized_datasets

output_dir = "from" + str(floor) + "to" + str(ceiling)

if not os.path.exists(output_dir):
    print("Directory does not exist. Creating it.")
    os.makedirs(output_dir)
else:
    print("Directory exists.")

print()

def insert_random_mask(batch):
    features = [dict(zip(batch, t)) for t in zip(*batch.values())]
    masked_inputs = whole_word_masking_data_collator(features)
    # Create a new "masked" column for each column in the dataset
    return {"masked_" + k: v.numpy() for k, v in masked_inputs.items()}

num_samples = min(2500, len(lm_datasets["first_test"]), len(lm_datasets["second_test"]))

test_dataset = concatenate_datasets(
    [lm_datasets["first_test"].shuffle(seed=42).select(range(num_samples)),
     lm_datasets["second_test"].shuffle(seed=42).select(range(num_samples))]
)

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

# Determine the number of samples to select
num_samples = min(20000, len(lm_datasets["first_train"]), len(lm_datasets["second_train"]))

train_dataset = concatenate_datasets(
    [lm_datasets["first_train"].shuffle(seed=42).select(range(num_samples)),
     lm_datasets["second_train"].shuffle(seed=42).select(range(num_samples))]
)

print('Training dataset length:', len(train_dataset))
print('Evaluation dataset length:', len(eval_dataset))

batch_size = 32
train_dataloader = DataLoader(
    train_dataset,
    shuffle=True,
    batch_size=batch_size,
    collate_fn=whole_word_masking_data_collator
)
eval_dataloader = DataLoader(
    eval_dataset, batch_size=batch_size, collate_fn=default_data_collator
)

optimizer = AdamW(model.parameters(), lr=5e-5)

# prepare model and data for acceleration

accelerator = Accelerator()
model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader
)

num_train_epochs = 64
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
    for batch in train_dataloader:
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
        
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(output_dir, save_function=accelerator.save)
        break