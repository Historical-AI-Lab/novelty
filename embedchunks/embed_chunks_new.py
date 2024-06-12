import os
import csv
import torch
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import math

# Function to generate embeddings given the directory with files in format paperID + .txt, with one chunk per line.
# Further down we'll open the folder and each file..

def generate_embeddings(texts, tokenizer, model, device):
    #If we need to use batches of chunks and loop through, they'll go up here.
    with torch.no_grad():
    # Tokenize and move to device
        batch_dict = tokenizer(texts, max_length=512, padding=True, truncation=True, return_tensors='pt')
        batch_dict = {k: v.to(device) for k, v in batch_dict.items()}

        # Generate embeddings
        outputs = model(**batch_dict)
        raw_embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
        embeddings = F.normalize(raw_embeddings, p=2, dim=1)

        # Explicitly delete tensors to free memory
        del batch_dict
        del outputs
        del raw_embeddings

        # Clear GPU cache
        torch.cuda.empty_cache()

    return embeddings

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

# Load tokenizer and model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using {device}')

tokenizer = AutoTokenizer.from_pretrained("thenlper/gte-base")
model = AutoModel.from_pretrained("thenlper/gte-base")
model.eval()
model.to(device)

# Add the directory with txt files here!
directory = 'clean-chunks-test0/'

results_for_output = []

# Here's where we're going through the directory to get the paperIDs, chunkIDs, and chunk text
# Add a print statement to monitor progress.
total_files = len(os.listdir(directory))
processed_files = 0

for file_name in os.listdir(directory):
    processed_files += 1
    if processed_files % 10 == 0:
        print(f"Processed {processed_files} out of {total_files} files.")

    if file_name.endswith('.txt'):
        paperID = file_name.split('.')[0]  
        file_path = os.path.join(directory, file_name)

        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) < 2:
                        print(f"Warning: Skipping line in {file_name}: {line} because fewer than 2 parts")
                        continue

                    #chunkID = f"{paperID}_{parts[0]}"  # The chunkID we'll use in our tsv file will have paperID + _ + 'tab' + chunk
                    chunkID = parts[0]
                    text = parts[1]

                    try:
                        embeddings = generate_embeddings(text, tokenizer, model, device)

                        # And check if embeddings are generated
                        if embeddings is not None and len(embeddings) > 0:
                            results_for_output.append([paperID, chunkID, embeddings])
                        else:
                            print(f"Warning: No embeddings generated for {chunkID}")
                    except Exception as e:
                        print(f"Error processing {chunkID}: {e}")

# Finally, for the output
output_tsv = 'output_embed9.tsv'

with open(output_tsv, 'w', newline='', encoding='utf-8') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerow(['paperID', 'chunkID', 'embeddings'])  
    for result in results_for_output:
        writer.writerow([result[0], result[1], ' '.join(map(str, result[2].cpu().numpy().tolist()))])  # Convert tensor to list

print(f"Embeddings have been saved to {output_tsv}")