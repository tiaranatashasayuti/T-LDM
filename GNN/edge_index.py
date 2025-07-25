from Bio.PDB import PDBParser, NeighborSearch
import pandas as pd
import os
import torch
from torch_geometric.utils import is_undirected
import os
from Bio.PDB import PDBParser

# Parameters
residue_mapping = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

def determine_edge_idx(record):
    print(f"{record.pdb_id}")
    STRUCTURE_FOLDER = "/home2/s230112/BIB_FINAL/GNN/docking_results"
    target_filename = f"{record.pdb_id.lower()}_emref_1.pdb"

    # Search all files in the directory, and match using lowercase
    all_files = os.listdir(STRUCTURE_FOLDER)
    matched_files = [f for f in all_files if f.lower() == target_filename]

    if matched_files:
        pdb_file_path = os.path.join(STRUCTURE_FOLDER, matched_files[0])
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("Complex", pdb_file_path)
    else:
        print("File Not Found")
        return "File Not Found", "File Not Found"


    distance_cutoff = 6.0  # Distance cutoff in Ångströms
    
    # Extract atoms
    chains = list(structure.get_chains())
    remove_list = []
    print(chains)
    for chain in chains:
        chain_residues = list(chain.get_residues())
        if len(chain_residues) > 2:  # Ensure the chain has more than two residues
            # Remove the first and last residues
            remove_list.append(chain_residues[0].id[1])
            chain.detach_child(chain_residues[0].id)
            chain.detach_child(chain_residues[-1].id)
            remove_list.append(chain_residues[-1].id[1])
          
    atoms = list(structure.get_atoms())
    # Identify ligand and receptor atoms
    source_atoms = [atom for atom in atoms if ((atom.get_parent().get_parent().id == "B" or atom.get_parent().get_parent().id == "A") and atom.element != "H")]
    target_atoms = [atom for atom in atoms if ((atom.get_parent().get_parent().id == "B" or atom.get_parent().get_parent().id == "A") and atom.element != "H")]
    
    # Perform neighbor search
    ns = NeighborSearch(target_atoms)
    interacting_residues = set()
    
    for src_atom in source_atoms:
        close_atoms = ns.search(src_atom.coord, distance_cutoff)  # Find nearby atoms within cutoff
        source_residue = src_atom.get_parent()  # Get the residue of the source atom
        chain_id = source_residue.get_parent().id
        residue_name = source_residue.get_resname()
        src_residue_id = source_residue.id[1]
        final_string_source = residue_mapping[str(residue_name)]+str(src_residue_id)
        for atom in close_atoms:
            target_residue = atom.get_parent()  # Get the residue of the target atom
            chain_id = target_residue.get_parent().id
            residue_name = target_residue.get_resname()
            tar_residue_id = target_residue.id[1]
            final_string_target = residue_mapping[str(residue_name)]+str(tar_residue_id)
            
            if target_residue.id[0] == " " and source_residue.id[0] == " " and final_string_source != final_string_target:  # Ignore hetero residues (e.g., water molecules)
                source_target_pair = (src_residue_id, tar_residue_id)
                interacting_residues.add(source_target_pair)
    
    source_list = []
    target_list = []
        
    sorted_removed = sorted(remove_list)
    for src_idx, tgt_idx in interacting_residues:
        # count how many removed residues come before each original index
        n_removed_before_src = sum(1 for r in sorted_removed if r < src_idx)
        n_removed_before_tgt = sum(1 for r in sorted_removed if r < tgt_idx)

        # subtract that count to re-index down to zero-based
        source_list.append(int(src_idx) - n_removed_before_src)
        target_list.append(int(tgt_idx) - n_removed_before_tgt)

    return source_list, target_list

BclxL_XLSX_FILE = "/home2/s230112/BIB_FINAL/GNN/input_features.csv"

complexes = pd.read_csv(BclxL_XLSX_FILE)
complexes.drop_duplicates(subset="pdb_id", inplace=True)
complexes.reset_index(drop=True, inplace=True)
rows = [complexes.iloc[i] for i in range(len(complexes))]

source_list, target_list = determine_edge_idx(rows[0])

return_list = {}
for i, row in enumerate(rows):
    source_list, target_list = determine_edge_idx(row)
    if source_list == "File Not Found":
        print(f"Skipping {row.pdb_id} as the file was not found.")
        continue

    edge_idx_array = [source_list, target_list]
    return_list[row.pdb_id] = edge_idx_array
    
OUTPUT_FILE = "/home2/s230112/BIB_FINAL/GNN/edge_indexes.pt"
torch.save(return_list, OUTPUT_FILE)