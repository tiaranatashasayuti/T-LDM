from Bio.PDB import PDBParser, NeighborSearch
import pandas as pd
import os
# Parameters

residue_mapping = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}

def extract_sequence_from_chain(chain):
    seq = []
    for residue in chain:
        if residue.id[0] == " ":
            resname = residue.get_resname()
            if resname in residue_mapping:
                seq.append(residue_mapping[resname])
    return ''.join(seq)
    
def determine_binding_site_row(record):
    print(f"{record.pdb_id}")
    STRUCTURE_FOLDER = "/home2/s230112/BIB_FINAL/GNN/docking_results/"# Folder contains all the bonded complexes
    target_filename = f"{record.pdb_id.upper()}_EMREF_1.PDB"

    # Loop through files in folder and match in case-insensitive manner
    matching_files = [
        f for f in os.listdir(STRUCTURE_FOLDER)
        if f.upper() == target_filename
    ]

    if matching_files:
        pdb_file_path = os.path.join(STRUCTURE_FOLDER, matching_files[0])
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("Complex", pdb_file_path)
    else:
        print("File Not Found")
        return "File Not Found", "File Not Found", "File Not Found", "File Not Found", "File Not Found"

    distance_cutoff = 6.0  # Distance cutoff in Ångströms
    receptor_chain = structure[0]['A']
    receptor_seq = extract_sequence_from_chain(receptor_chain)

    # Extract atoms
    atoms = list(structure.get_atoms())

    # Identify ligand and receptor atoms
    ligand_atoms = [atom for atom in atoms if atom.get_parent().get_parent().id == "B" and atom.element != "H"]
    receptor_atoms = [atom for atom in atoms if atom.get_parent().get_parent().id == "A" and atom.element != "H"]

    # Perform neighbor search
    ns = NeighborSearch(receptor_atoms)
    binding_site_residues = set()

    for ligand_atom in ligand_atoms:
        close_atoms = ns.search(ligand_atom.coord, distance_cutoff)  # Find nearby atoms within cutoff
        for atom in close_atoms:
            residue = atom.get_parent()  # Get the residue of the receptor atom
            if residue.id[0] == " ":  # Ignore hetero residues (e.g., water molecules)
                binding_site_residues.add(residue)
    
    if len(binding_site_residues) == 0:
        print("No close residues")
        return "No close residues", "No close residues", "No close residues", "No close residues", "No close residues"
    # Output the binding site residues
    RBS_site = []
    LBS_site = []

    for residue in binding_site_residues:
        #chain_id = residue.get_parent().id
        residue_name = residue.get_resname()
        residue_id = residue.id[1]
        final_string = residue_mapping[str(residue_name)]+str(residue_id)
        RBS_site.append(final_string)      

    RBS_site_seq = ' '.join(RBS_site)
    print(RBS_site_seq)

    ns = NeighborSearch(ligand_atoms)
    binding_site_residues = set()

    for receptor_atom in receptor_atoms:
        close_atoms = ns.search(receptor_atom.coord, distance_cutoff)  # Find nearby atoms within cutoff
        for atom in close_atoms:
            residue = atom.get_parent()  # Get the residue of the receptor atom
            binding_site_residues.add(residue)

    # Output the binding site residues
    for residue in binding_site_residues:
        #chain_id = residue.get_parent().id
        residue_name = residue.get_resname()
        residue_id = residue.id[1]
        final_string = residue_mapping[str(residue_name)]+str(residue_id)
        LBS_site.append(final_string)
    LBS_site_seq = ' '.join(LBS_site)
    print(LBS_site_seq)
    return "A", "B", receptor_seq, RBS_site_seq, LBS_site_seq



BclxL_XLSX_FILE = "/home2/s230112/BIB_FINAL/GNN/peptide_data.csv"

complexes = pd.read_csv(BclxL_XLSX_FILE)
complexes.drop_duplicates(subset="pdb_id", inplace=True)
complexes.reset_index(drop=True, inplace=True)
rows = [complexes.iloc[i] for i in range(len(complexes))]
complexes["receptor_chain"] = ""
complexes["ligand_chain"] = ""
complexes["receptor_seq"] = ""
complexes["binding_site_pdb"] = ""
complexes["LBS_site"] = ""

for i, row in enumerate(rows):
    r_chain, l_chain, r_seq, binding_site, LBS_site = determine_binding_site_row(row)
    complexes.loc[i, "receptor_chain"] = r_chain
    complexes.loc[i, "ligand_chain"] = l_chain
    complexes.loc[i, "receptor_seq"] = r_seq
    complexes.loc[i, "LBS_site"] = LBS_site
    complexes.loc[i, "binding_site_pdb"] = binding_site

#print(complexes)
complexes.to_csv("input_features.csv")