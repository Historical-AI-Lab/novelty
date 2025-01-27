# LoadTimeSlice
#
# This function accepts a floor, a ceiling, and a metadata dataframe.
# It selects paperIds between the floor and ceiling, and loads the text
# files corresponding to those paperIds. It transforms them first into
# a pandas dataframe and then into a HuggingFace DatasetDict, which
# it returns.

import pandas as pd
from datasets import Dataset, DatasetDict

metadata = pd.read_csv('/projects/ischoolichass/ichass/usesofscale/novelty/metadata/litstudies/LitMetadataWithS2.tsv', sep = '\t')
metadata['year'] = metadata['year'].astype(int)

# Drop rows with missing paperId values
metadata = metadata.dropna(subset=['paperId'])

# Filter out paperIds with length less than 2
metadata = metadata[metadata['paperId'].str.len() > 2]

def LoadTimeSlice(floor, ceiling, metadata):
    selected_paperIds = metadata[(metadata['year'] >= floor) & (metadata['year'] <= ceiling)]['paperId']

    rootfolder = '/projects/ischoolichass/ichass/usesofscale/novelty/embeddingcode/chunks'

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

minyear = min(metadata['year'])
maxyear = max(metadata['year'])

assert isinstance(minyear, int), "minyear must be an integer"

for floor in range(minyear, maxyear - 12, 4):
    ceiling = floor + 12
    dataset = LoadTimeSlice(floor, ceiling, metadata)
    print('successfully loaded dataset for', floor, 'to', ceiling)
    break