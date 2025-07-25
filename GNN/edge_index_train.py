from Bio.PDB import PDBParser, NeighborSearch
import pandas as pd
import os
import torch
from torch_geometric.utils import is_undirected

# Parameters

residue_mapping = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def determine_edge_idx(record, receptor_chain_id, ligand_chain_id):
    id_val = record["pdb_id"]
    print(f"{id_val.values[0]}")
    STRUCTURE_FOLDER = "/home2/s230112/BIB_FINAL/GNN/train_data"# Folder contains all the bonded complexes
    pdb_file_path = os.path.join(STRUCTURE_FOLDER, f"{id_val.values[0]}.pdb")
    if os.path.exists(pdb_file_path):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("Complex", pdb_file_path)
    else:
        print("File Not Found")
        return "File Not Found", "File Not Found"

    distance_cutoff = 6.0  # Distance cutoff in Ångströms
    
    # Extract atoms
    chain_residuo = []
    chains = list(structure.get_chains())
    remove_list = []
    monitor_set = set()
    mapping_dict = {}
    for chain in chains:
        if chain.get_id() not in [receptor_chain_id, ligand_chain_id]:
            continue
        if chain in monitor_set:
            continue
        monitor_set.add(chain)
        if len(monitor_set) == 2:
            break
    ordered_monitor_set = list()
    for item in monitor_set:
        if item.get_id() == receptor_chain_id:
            ordered_monitor_set.insert(0,item)
        elif item.get_id() == ligand_chain_id:
            ordered_monitor_set.insert(-1,item)

    i = -1
    k = 0
    while k < len(ordered_monitor_set):
        chain = ordered_monitor_set[k]
        chain_residues = list(chain.get_residues())
        for resido in chain_residues: 
            if resido.id[0] == ' ':
                if str(resido.get_resname()) not in residue_mapping.keys():
                    continue
                res_name = residue_mapping[str(resido.get_resname())]
                full_name = res_name + str(resido.id[1]) + str(chain.get_id())
                chain_residuo.append(resido)
                mapping_dict[full_name] = i
                i += 1

        if len(chain_residuo) > 2:  # Ensure the chain has more than two residues
            # Remove the first and last residues
            zero_res_name = residue_mapping[str(chain_residuo[0].get_resname())]
            zero_name = zero_res_name + str(chain_residuo[0].id[1]) + str(chain.get_id())
            if zero_name not in remove_list:
                remove_list.append(zero_name)
                chain.detach_child(chain_residuo[0].id)
            #print(chain_residues[-1].id)
            last_res_name = residue_mapping[str(chain_residuo[-1].get_resname())]
            last_name = last_res_name + str(chain_residuo[-1].id[1]) + str(chain.get_id())
            if last_name not in remove_list:
                chain.detach_child(chain_residuo[-1].id)
                remove_list.append(last_name)
        chain_residuo = []
        i -= 2
        k += 1
    atoms = list(structure.get_atoms())
    # Identify ligand and receptor atoms
    source_atoms = [atom for atom in atoms if ((atom.get_parent().get_parent().id == ligand_chain_id or atom.get_parent().get_parent().id == receptor_chain_id) and atom.element != "H" and atom.get_parent().id[0] == " ")]
    target_atoms = [atom for atom in atoms if ((atom.get_parent().get_parent().id == ligand_chain_id or atom.get_parent().get_parent().id == receptor_chain_id) and atom.element != "H" and atom.get_parent().id[0] == " ")]
    
    # Perform neighbor search
    ns = NeighborSearch(target_atoms)
    interacting_residues = set()
    
    for src_atom in source_atoms:
        close_atoms = ns.search(src_atom.coord, distance_cutoff)  # Find nearby atoms within cutoff
        source_residue = src_atom.get_parent()  # Get the residue of the source atom
        chain_id = source_residue.get_parent().id
        residue_name = source_residue.get_resname()
        src_residue_id = source_residue.id[1]
        if source_residue.id[0] != " " or str(residue_name) not in residue_mapping.keys():
            continue
        final_string_source = residue_mapping[str(residue_name)]+str(src_residue_id)+str(chain_id)
        
        for atom in close_atoms:
            target_residue = atom.get_parent()  # Get the residue of the target atom
            chain_id = target_residue.get_parent().id
            residue_name = target_residue.get_resname()
            tar_residue_id = target_residue.id[1]
            if target_residue.id[0] != " " or str(residue_name) not in residue_mapping.keys():
                continue
            final_string_target = residue_mapping[str(residue_name)]+str(tar_residue_id)+str(chain_id)
            
            if target_residue.id[0] == " " and source_residue.id[0] == " " and final_string_source != final_string_target:  # Ignore hetero residues (e.g., water molecules)
                source = mapping_dict[final_string_source]
                target = mapping_dict[final_string_target]
                if (final_string_source in remove_list) or (final_string_target in remove_list):
                    continue
                source_target_pair = (source, target)
                interacting_residues.add(source_target_pair)
    
    source_list = []
    target_list = []
    for pair in interacting_residues:
        source_list.append(int(pair[0]))
        target_list.append(int(pair[1]))
    return source_list, target_list

    

file_path = "biolip.pt"
data_features = torch.load(file_path)

BIOLIP_META_HEADER = [
    "pdb_id",
    "receptor_chain",
    "resolution",
    "binding_site",
    "ligand_ccd_id",
    "ligand_chain",
    "ligand_serial_num",
    "binding_site_pdb", 
    "binding_site_reorder",
    "catalyst_site_pdb",
    "catalyst_site_reorder",
    "enzyme_class_id",
    "go_term_id",
    "binding_affinity_literature",
    "binding_affinity_binding_moad",
    "binding_affinity_pdbind_cn",
    "binding_affinity_binding_db",
    "uniprot_db",
    "pubmed_id",
    "ligand_res_num",
    "receptor_seq"
]
BIOLIP_META_FILE = "/home2/s230112/BIB_FINAL/GNN/BioLiP.txt"
complexes = pd.read_csv(BIOLIP_META_FILE, sep="\t", names=BIOLIP_META_HEADER)
complexes.drop_duplicates(subset="pdb_id", inplace=True) # VERY IMPORTANT
complexes.reset_index(drop=True, inplace=True)
complexes = complexes.loc[complexes.resolution<5] # Selecting complexes which have a resolution of less than 5
return_list = {}

for val in data_features:
    #pdb_id = val['structure_ids']['pdb_id']

    #tiara edit
    pdb_id = val['structure_ids']['pdb_id'].strip() # clean pdb id

    print(f"Processing pdb_id: {pdb_id}")
    
    if pdb_id not in complexes['pdb_id'].values:
        print(f"pdb_id {pdb_id} not found in complexes.")
        continue

    selected_row = complexes[complexes['pdb_id'] == pdb_id]
    
    #tiara edit
    if selected_row.empty:
        print(f"No matching row found for pdb_id: {pdb_id}")
        continue
    
    source_list, target_list = determine_edge_idx(selected_row, selected_row["receptor_chain"].values[0], selected_row["ligand_chain"].values[0])
    if source_list == "File Not Found":
        continue
    edge_idx_array = [source_list, target_list]
    return_list[selected_row["pdb_id"].values[0]] = edge_idx_array
    print("Done")

OUTPUT_FILE = "edge_indexes_train_test.pt"
torch.save(return_list, OUTPUT_FILE)