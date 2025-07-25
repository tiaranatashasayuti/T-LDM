import torch
file_path = "/home2/s230112/BIB_FINAL/GNN/biolip.pt"
data_features = torch.load(file_path)

import os
import requests
from time import sleep


# Directory to save files
output_dir = "/home2/s230112/BIB_FINAL/GNN/train_data"

# Download each PDB file
for val in data_features:
    pdb_id = val['structure_ids']['pdb_id']
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    output_path = os.path.join(output_dir, f"{pdb_id}.pdb")
    
    # Skip if already downloaded
    if os.path.exists(output_path):
        print(f"{pdb_id}.pdb already exists, skipping.")
        continue
    
    # Download the file
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise error for bad responses
        with open(output_path, "w") as file:
            file.write(response.text)
        print(f"Downloaded {pdb_id}.pdb")
    except Exception as e:
        print(f"Failed to download {pdb_id}: {e}")
        sleep(5)  # Avoiding rapid retries
