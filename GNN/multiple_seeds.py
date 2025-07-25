from train_GCN import set_seed, BindingPocketGNN, FocalLoss, ProteinLigandDataset, create_graph
# from train_GAT_no_coords import set_seed, BindingPocketGNN, FocalLoss, ProteinLigandDataset, create_graph
from torch_geometric.loader import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import os

# Load data for reuse across all seeds
features = torch.load("features.pt", weights_only=True)
raw_edges = torch.load("edge_indexes.pt", weights_only=True)
edge_indexes = {k.upper(): v for k, v in raw_edges.items()}
features = [f for f in features if f['structure_ids']['pdb_id'].upper() in edge_indexes]

graph_data_list = []
for complex in features:
    graph = create_graph(complex, edge_indexes)
    if graph != "No Graph":
        graph_data_list.append(graph)

# Split dataset
total = len(graph_data_list)
train_size = int(0.7 * total)
val_size = int(0.15 * total)
train_data = graph_data_list[:train_size]
val_data = graph_data_list[train_size:train_size + val_size]
test_data = graph_data_list[train_size + val_size:]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Train and save for each seed
for seed in range(41, 51):
    print(f"\n🔁 Training with seed {seed}")
    set_seed(seed)

    model = BindingPocketGNN(input_dim=15, hidden_dim=256, output_dim=1).to(device)
    loss_fn = FocalLoss(alpha=1.0, gamma=2.5)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    train_loader = DataLoader(ProteinLigandDataset(train_data), batch_size=64, shuffle=True)
    val_loader = DataLoader(ProteinLigandDataset(val_data), batch_size=64, shuffle=False)

    for epoch in range(70):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            logits, probs = model(batch)
            loss = loss_fn(logits, batch.y.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # save GAT model        
    # torch.save(model.state_dict(), f"model_finetuned_seed_{seed}.pth")

    #save GCN model 
    torch.save(model.state_dict(), f"model_GCN_finetuned_seed_{seed}.pth")
    print(f"✅ Saved: model_finetuned/GCN_seed_{seed}.pth")
