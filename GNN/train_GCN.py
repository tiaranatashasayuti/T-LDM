###############################################
#           IMPORTS AND SETUP                 #
###############################################
from torch_geometric.data import Dataset, Data
import torch
from torch_geometric.loader import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
import numpy as np
from torch_geometric.utils import is_undirected, subgraph
from sklearn.metrics import classification_report, roc_auc_score
from torch.optim.lr_scheduler import StepLR
import wandb  
import argparse  
import random
import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.set_num_threads(4) 
from sklearn.metrics import precision_score, recall_score

def set_seed(seed_1):
    random.seed(seed_1) 
    np.random.seed(seed_1)  
    torch.manual_seed(seed_1)  
    torch.cuda.manual_seed(seed_1)  
    torch.cuda.manual_seed_all(seed_1)  
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  
    torch.use_deterministic_algorithms(True, warn_only=True) 

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


###############################################
#           HELPER FUNCTIONS                  #
###############################################

# --- Encoding functions ---
secondary_structure_dict = {}
def encoding_categorical_data(data):
    main_list = []
    for cat in data:
        if cat in secondary_structure_dict:
            main_list.append(secondary_structure_dict[cat])
        else:
            if len(secondary_structure_dict.values()) == 0:
                secondary_structure_dict[cat] = 0
            else:
                secondary_structure_dict[cat] = len(secondary_structure_dict)
            main_list.append(secondary_structure_dict[cat])
    return main_list

def encoding_masks(data):
    # Convert boolean mask to binary (1/0)
    return [1 if val else 0 for val in data]

def encoding_amino(data):
    residue_mapping = {
        "A": 0, "C": 4, "D": 8, "E": 12, "F": 16,
        "G": 1, "H": 5, "I": 9, "K": 13, "L": 17,
        "M": 2, "N": 6, "P": 10, "Q": 14, "R": 18,
        "S": 3, "T": 7, "V": 11, "W": 15, "Y": 19,
    }
    return [residue_mapping[val] for val in data]

# --- Graph construction ---
def create_graph(data, edge_indexes_input):
    pdb_id = data['structure_ids']['pdb_id'].upper()
    if pdb_id not in edge_indexes_input:
        print(f"[create_graph] no edges for {pdb_id}, skipping")
        return "No Graph"
    
    # Lookup raw edge list (two lists of indices)
    raw_edges = edge_indexes_input[pdb_id]
    src = torch.tensor(raw_edges[0], dtype=torch.int64)
    tgt = torch.tensor(raw_edges[1], dtype=torch.int64)
    
    # Encode the features
    secondary_structure_encoded = encoding_categorical_data(data['secondary_structure'])
    pocket_mask_encoded = encoding_masks(data['pocket_mask'])
    acid_encoded = encoding_amino(data['amino_acid'])

    # Concatenate features into a node feature tensor:
    node_features = torch.cat([
        data['numerical_features'],                                    # Numerical features
        torch.tensor(secondary_structure_encoded).unsqueeze(1),        # Encoded secondary structure 
        #data['coors'],                                                 # Coordinates
        torch.tensor(acid_encoded).unsqueeze(1),                       # Encoded amino acid 
        data['angle_features']                                        # Angle features
    ], dim=1).to(torch.float32)
    
    num_nodes = node_features.size(0)
    
    # Filter out any edges touching a missing node
    valid_mask = (src < num_nodes) & (tgt < num_nodes)
    edge_index_val = torch.stack([src[valid_mask], tgt[valid_mask]], dim=0)
    # If after filtering there are no edges at all, skip
    if edge_index_val.size(1) == 0:
        return "No Graph"
    
    # use only the kept (valid_mask) edges for expansion
    tuple_list = list(zip(
    src[valid_mask].tolist(),
    tgt[valid_mask].tolist(),
))
    # Build the initial graph with node features, edge indices, and node labels
    graph = Data(
        x=node_features,
        edge_index=edge_index_val,
        y=torch.tensor(pocket_mask_encoded).unsqueeze(1)
    )
    
    # Extract a subgraph focused on the pocket
    to_keep_nodes = set(data['pocket_idx'].tolist())
    in_progress_nodes = set(to_keep_nodes)
    for src, tgt in tuple_list:
        if src in to_keep_nodes:
            in_progress_nodes.add(tgt)

    # keep these nodes
    subset_nodes = torch.tensor(sorted(in_progress_nodes), dtype=torch.long)

    # Debug 
    if False:
        print(f"PDB: {data['structure_ids']['pdb_id']}")
        print(f"Node feature count: {num_nodes}")
        print(f"Pocket idx max: {max(to_keep_nodes) if to_keep_nodes else 'None'}")
        print(f"Expanded idx max: {max(in_progress_nodes) if in_progress_nodes else 'None'}")
        print(f"Number of expanded nodes: {len(in_progress_nodes)}")
        print(f"Edge_index max: {int(graph.edge_index.max())}, min: {int(graph.edge_index.min())}")

    # Safe subgraph with the full node count
    new_edge_index, _ = subgraph(
        subset_nodes,
        graph.edge_index,
        relabel_nodes=True,
        num_nodes=num_nodes
    )

    # Set ligand indices to 1 
    for idx in data["ligand_idx"].tolist():
        if idx < num_nodes:
            graph.y[idx] = 1

    new_x = graph.x[subset_nodes]
    new_y = graph.y[subset_nodes]
    
    return Data(x=new_x, edge_index=new_edge_index, y=new_y)

# --- Dataset Class ---
class ProteinLigandDataset(Dataset):
    def __init__(self, graph_data_list, transform=None):
        super().__init__(transform)
        self.graph_data_list = graph_data_list
        
    def __len__(self):
        return len(self.graph_data_list)

    def __getitem__(self, idx):
        return self.graph_data_list[idx]

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

###############################################
#            MODEL DEFINITION                 #
###############################################
class BindingPocketGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BindingPocketGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.2)
        self.fc = torch.nn.Linear(hidden_dim, output_dim)
        
    def forward(self, data, return_embeddings=False):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = self.dropout(torch.relu(x))

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = self.dropout(torch.relu(x))
        # Get latent embeddings
        latent_embeddings = torch.relu(self.conv3(x, edge_index))
        out = self.fc(latent_embeddings)
        if return_embeddings:
            return latent_embeddings, torch.sigmoid(out)
        
        return out, torch.sigmoid(out)

###############################################
#          TRAINING & EVALUATION LOOPS        #
###############################################
def train_loop(model, loader, loss_fn, optimizer, device):
    model.train()
    running_loss, correct = 0.0, 0
    threshold = 0.5
    num_nodes = 0
    for batch in loader:
        batch = batch.to(device)
        num_nodes += batch.num_nodes
        optimizer.zero_grad()
        logits, probabilities = model(batch)
        predictions = (probabilities > threshold)
        loss = loss_fn(logits, batch.y.float().to(device))
        correct += (predictions == batch.y.to(device)).float().sum().item()
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch.num_nodes
    return running_loss / len(loader.dataset), correct / num_nodes

def valid_loop(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct = 0.0, 0
    threshold = 0.5
    num_nodes = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes += batch.num_nodes
            logits, probabilities = model(batch)
            predictions = (probabilities > threshold)
            loss = loss_fn(logits, batch.y.float().to(device))
            correct += (predictions == batch.y.to(device)).float().sum().item()
            running_loss += loss.item() * batch.num_nodes
    return running_loss / len(loader.dataset), correct / num_nodes

def test_loop(model, loader, loss_fn, device):
    model.eval()
    running_loss, correct = 0.0, 0
    threshold = 0.5
    num_nodes = 0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            num_nodes += batch.num_nodes
            logits, probabilities = model(batch)
            predictions = (probabilities > threshold)
            loss = loss_fn(logits, batch.y.float().to(device))
            correct += (predictions == batch.y.to(device)).float().sum().item()
            running_loss += loss.item() * batch.num_nodes

            # Accumulate predictions and labels
            all_preds.append(predictions.cpu())
            all_labels.append(batch.y.cpu())
            all_probs.append(probabilities.cpu())

    # Concatenate all batches
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    all_probs = torch.cat(all_probs)

    # Print final combined report
    print(classification_report(all_labels, all_preds))
    auc = roc_auc_score(all_labels, all_probs)
    print(f"AUC-ROC: {auc:.4f}")

    return running_loss / len(loader.dataset), correct / num_nodes


def extract_embedding(model, embeddings_loader, device, dataset_name):
    model.eval()
    store_embeddings = []
    with torch.no_grad():
        for batch in embeddings_loader:
            batch = batch.to(device)
            batch_embeddings, y_prediction_probabilities = model(batch, return_embeddings=True)
            graph_embedding = global_mean_pool(batch_embeddings, batch.batch)
            store_embeddings.append(graph_embedding)
    
    concat_result = torch.cat(store_embeddings, dim=0)
    results_dict = {}
    
    valid_pdb_ids = valid_acp_pdb_ids if dataset_name == "acp" else valid_train_pdb_ids

    # Ensure the lengths match before assignment
    assert len(concat_result) == len(valid_pdb_ids), \
        f"Mismatch: {len(concat_result)} embeddings vs. {len(valid_pdb_ids)} valid PDB IDs"

    results_dict = {valid_pdb_ids[i]: concat_result[i] for i in range(len(concat_result))}


    output_file = f"resulting_embeddings_GCN_{dataset_name}.pt"
    torch.save(results_dict, output_file)
    print(f"Embeddings saved to {output_file}")

def extract_embedding_pocket(model, embeddings_loader,device, dataset_name):
    model.eval()
    store_embeddings = []
    threshold = 0.5
    with torch.no_grad():
        for bonded_complexes in embeddings_loader:  
            bonded_complexes = bonded_complexes.to(device) 
            batch_embeddings, y_prediction_probabilities =  model(bonded_complexes, return_embeddings = True)
            y_val = bonded_complexes.y
            pocket_mask = torch.squeeze(y_val == 1)
            pocket_batch_embeddings = batch_embeddings[pocket_mask]
            pocket_batch = bonded_complexes.batch[pocket_mask]
            graph_embedding = global_mean_pool(pocket_batch_embeddings, pocket_batch)
            store_embeddings.append(graph_embedding)  
            
    concat_result = torch.cat(store_embeddings, dim=0)
    results_dict = {}
    
    valid_pdb_ids = valid_acp_pdb_ids if dataset_name == "acp" else valid_train_pdb_ids

    # Ensure the lengths match before assignment
    assert len(concat_result) == len(valid_pdb_ids), \
        f"Mismatch: {len(concat_result)} embeddings vs. {len(valid_pdb_ids)} valid PDB IDs"

    results_dict = {valid_pdb_ids[i]: concat_result[i] for i in range(len(concat_result))}


    output_file = f"resulting_embeddings_GCN_pocket_{dataset_name}.pt"
    torch.save(results_dict, output_file)
    print(f"Embeddings saved to {output_file}")


###############################################
#             DATA LOADING                    #
###############################################
# --- Load ACP (BCL-xL) data ---
file_path = "features.pt"
edge_path = "edge_indexes.pt"
features = torch.load(file_path, weights_only=True)
raw_ei = torch.load(edge_path, weights_only=True)
id_to_test= features[0]['structure_ids']['pdb_id']
edge_indexes = {k.upper(): v for k, v in raw_ei.items()}

valid_keys = {c['structure_ids']['pdb_id'].upper() for c in features}
edge_indexes = { k: v for k, v in edge_indexes.items() if k in valid_keys }
raw_keys_upper = { k.upper() for k in raw_ei.keys() }
missing = sorted(valid_keys - raw_keys_upper)
print(f"These PDB IDs are in features.pt but not in edge_indexes.pt:\n{missing}")

features = [f for f in features
            if f['structure_ids']['pdb_id'].upper() in edge_indexes]

pdb_ids   = [c['structure_ids']['pdb_id'].upper() for c in features]
edge_keys = list(edge_indexes.keys())

print("Total features:", len(pdb_ids))
print("Total edge-keys:", len(edge_keys))
print("Intersection size:", len(set(pdb_ids) & set(edge_keys)))
print("Missing examples:", list(set(pdb_ids) - set(edge_keys)))
print("Extra edge-keys:", list(set(edge_keys) - set(pdb_ids)))

print("Loading biolip")
# --- Load Biolip data (for pretraining) ---
train_file_path = "biolip.pt"
train_edge_path = "edge_indexes_train.pt"
train_features = torch.load(train_file_path, weights_only=True)
train_edge_raw = torch.load(train_edge_path, weights_only=True)
train_edge_indexes = { k.upper(): v for k, v in train_edge_raw.items() }

train_features = [
    c for c in train_features
    if c['structure_ids']['pdb_id'].upper() in train_edge_indexes
]

# --- Build graph lists ---
# Pretraining graphs (Biolip)
train_graph_data_list = []
valid_train_pdb_ids = []  # Store PDB IDs for valid graphs

for complex in train_features:
    pdb = complex['structure_ids']['pdb_id'].upper()
    if pdb not in train_edge_indexes:
        continue
    graph = create_graph(complex, train_edge_indexes)
    if graph == "No Graph":
        continue
    train_graph_data_list.append(graph)
    valid_train_pdb_ids.append(complex['structure_ids']['pdb_id'])

# Fine-tuning graphs (ACP / BCL-xL)
graph_data_list = []
valid_acp_pdb_ids = []  # Store PDB IDs for valid graphs

empty_graphs=[]

for complex in features:
    i=0
    graph = create_graph(complex, edge_indexes)
    if graph == "No Graph":
        print("Detected",i)
        i+=1
        continue
    graph_data_list.append(graph)
    valid_acp_pdb_ids.append(complex['structure_ids']['pdb_id'])  # Keep track of valid IDs

empty_count = 0
for i, g in enumerate(graph_data_list):
    num_nodes = g.x.size(0)
    num_edges = g.edge_index.size(1)
    if num_nodes == 0 or num_edges == 0:
        empty_count += 1
        empty_graphs.append(valid_acp_pdb_ids[i])

    if i < 5:
        print(f"  graph[{i}]:  nodes={num_nodes}, edges={num_edges}")
print(f"→ {empty_count}/{len(graph_data_list)+empty_count} graphs are empty (zero nodes or zero edges)")

def get_test_env():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    loss_fn = FocalLoss(alpha=1.0, gamma=2.5)

    features = torch.load("features.pt", weights_only=True)
    edge_indexes_raw = torch.load("edge_indexes.pt", weights_only=True)
    edge_indexes = {k.upper(): v for k, v in edge_indexes_raw.items()}
    features = [f for f in features if f['structure_ids']['pdb_id'].upper() in edge_indexes]

    graph_data_list = []
    for complex in features:
        graph = create_graph(complex, edge_indexes)
        if graph != "No Graph":
            graph_data_list.append(graph)

    total = len(graph_data_list)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    test_data = graph_data_list[train_size + val_size:]

    test_loader = DataLoader(ProteinLigandDataset(test_data), batch_size=64, shuffle=False)
    return device, loss_fn, test_loader

print("Proceeding to main")

###############################################
#           MAIN FUNCTION (with wandb)        #
###############################################
def main():
    # Parse command-line arguments to choose the GPU.
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3, help="GPU device id to use (e.g., 0, 1, 2, or 3)")
    args, unknown = parser.parse_known_args()
    args.gpu = 0
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    set_seed(42)

    g = torch.Generator()
    g.manual_seed(42)
    
    # Initialize wandb; hyperparameters are set via wandb.config (sweep will override these)
    wandb.init(project="GNN_ProteinLigand", config={
        "hidden_size": 256,
        "pretrain_epochs": 1,
        "finetune_epochs": 1,
        "focal_alpha": 1,
        "focal_gamma": 2.5,
        "lr_pretrain": 0.0005,
        "lr_finetune": 0.0005
    })
    config = wandb.config


    ###############################################
    #           PHASE 1: PRETRAINING              #
    ###############################################
    split_idx = int(0.8 * len(train_graph_data_list))
    pretrain_train_dataset = ProteinLigandDataset(train_graph_data_list[:split_idx])
    pretrain_val_dataset = ProteinLigandDataset(train_graph_data_list[split_idx:])

    pretrain_train_loader = DataLoader(pretrain_train_dataset, batch_size=64, shuffle=True, worker_init_fn=seed_worker, generator=g)
    pretrain_val_loader = DataLoader(pretrain_val_dataset, batch_size=64, shuffle=False)

    ###############################################
    #  Initialize Model, Loss & Optimizer (Pretrain)
    ###############################################
    in_channels = 15 
    out_channels = 1
    hidden_size = config.hidden_size
    model = BindingPocketGNN(in_channels, hidden_size, out_channels)
    model = model.to(device)

    # To do pretraining
    optimizer = optim.Adam(model.parameters(), lr=config.lr_pretrain, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)
    loss_fn = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    
    pretrain_epochs = config.pretrain_epochs
    print("----- Pretraining on Biolip Data -----")
    for epoch in range(pretrain_epochs):
        train_loss, train_acc = train_loop(model, pretrain_train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = valid_loop(model, pretrain_val_loader, loss_fn, device)
        scheduler.step() 
        
        print(f"[Pretrain] Epoch {epoch+1}/{pretrain_epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        wandb.log({
            "pretrain/train_loss": train_loss,
            "pretrain/train_acc": train_acc,
            "pretrain/val_loss": val_loss,
            "pretrain/val_acc": val_acc,
            "epoch": epoch+1
        })

    # Save the pretrained model state (optional)
    torch.save(model.state_dict(), 'model_pretrained_coordinate_abalation.pth')

    ###############################################
    #          PHASE 2: FINE-TUNING               #
    ###############################################
    # For the ACP (BCL-xL) data, we split the dataset 70:15:15.
    total = len(graph_data_list)
    train_size = int(0.7 * total)
    val_size = int(0.15 * total)
    # Test size is the remainder.
    test_size = total - train_size - val_size

    print("Total ACP examples:", total)
    print("Train size:", train_size, "Validation size:", val_size, "Test size:", test_size)

    finetune_train_dataset = ProteinLigandDataset(graph_data_list[:train_size])
    finetune_val_dataset   = ProteinLigandDataset(graph_data_list[train_size:train_size+val_size])
    finetune_test_dataset  = ProteinLigandDataset(graph_data_list[train_size+val_size:])

    finetune_train_loader = DataLoader(finetune_train_dataset, batch_size=64, shuffle=False, worker_init_fn=seed_worker, generator=g)
    finetune_val_loader = DataLoader(finetune_val_dataset, batch_size=64, shuffle=False)
    finetune_test_loader = DataLoader(finetune_test_dataset, batch_size=64, shuffle=False)

    # Optionally, use a lower learning rate for fine-tuning.
    optimizer = optim.Adam(model.parameters(), lr=config.lr_finetune, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = FocalLoss(alpha=config.focal_alpha, gamma=config.focal_gamma)

    finetune_epochs = config.finetune_epochs
    print("----- Fine-tuning on ACP (BCL-xL) Data -----")
    for epoch in range(finetune_epochs):
        train_loss, train_acc = train_loop(model, finetune_train_loader, loss_fn, optimizer, device)
        val_loss, val_acc = valid_loop(model, finetune_val_loader, loss_fn, device)
        scheduler.step(val_loss)
        print(f"[Finetune] Epoch {epoch+1}/{finetune_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")
        wandb.log({
            "finetune/train_loss": train_loss,
            "finetune/train_acc": train_acc,
            "finetune/val_loss": val_loss,
            "finetune/val_acc": val_acc,
            "finetune_epoch": epoch+1
        })

    # Evaluate on the fine-tuning test set
    test_loss, test_acc = test_loop(model, finetune_test_loader, loss_fn, device)
    print(f"Fine-tuning Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc*100:.2f}%")
    wandb.log({
        "final_test_loss": test_loss,
        "final_test_acc": test_acc
    })
    # Also record the final test accuracy in the summary so the sweep can pick it up.
    wandb.run.summary["final_test_acc"] = test_acc

    # Save the final fine-tuned model
    torch.save(model.state_dict(), 'model_finetuned_coordinate_abalation.pth')

    ###############################################
    #         EMBEDDING EXTRACTION                #
    ###############################################

    # Extract embeddings separately for Biolip (pretraining dataset)
    biolip_dataset = ProteinLigandDataset(train_graph_data_list)
    biolip_loader = DataLoader(biolip_dataset, batch_size=64, shuffle=False)

    extract_embedding(model, biolip_loader, device, dataset_name="biolip")
    extract_embedding_pocket(model, biolip_loader, device, dataset_name="biolip")

    # Extract embeddings separately for ACP-BCL-xL (fine-tuning dataset)
    acp_dataset = ProteinLigandDataset(graph_data_list)
    acp_loader = DataLoader(acp_dataset, batch_size=64, shuffle=False)
    extract_embedding(model, acp_loader, device, dataset_name="acp")
    extract_embedding_pocket(model, acp_loader, device, dataset_name="acp")


    wandb.finish()

if __name__ == '__main__':
    main()