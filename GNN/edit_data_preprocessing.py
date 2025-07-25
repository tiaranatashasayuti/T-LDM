import os
import torch
import warnings
import itertools
import numpy as np
import pandas as pd
from Bio.PDB import DSSP
from Bio.PDB.Chain import Chain
from Bio.PDB import PDBParser

warnings.filterwarnings("ignore", module="Bio")

OUTPUT_FILE = "features.pt"
STRUCTURE_FOLDER = "/home2/s230112/BIB_FINAL/GNN/docking_results/"# Folder contains all the bonded complexes
BclxL_XLSX_FILE = "/home2/s230112/BIB_FINAL/GNN/input_features.csv"



def calc_angle(p1,p2,p3):
    v1 = p2 - p1
    v2 = p2 - p3
    dot_product = np.dot(v1, v2)
    mag_v1 = np.linalg.norm(v1)
    mag_v2 = np.linalg.norm(v2)
    if mag_v1 == 0 or mag_v2 == 0:
        raise ValueError("One of the vectors has zero magnitude, leading to an undefined angle.")
    cos_theta = dot_product / (mag_v1 * mag_v2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # avoid float precision problem
    theta = np.arccos(cos_theta)
    theta_degrees = np.degrees(theta)
    return theta_degrees

def calc_dihedral(p1, p2, p3, p4):
    # Convert points to numpy arrays
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)
    # Vectors
    v1 = p2 - p1
    v2 = p3 - p2
    v3 = p4 - p3
    # Normal vectors
    n1 = np.cross(v1, v2)  # Normal to the plane formed by p1, p2, p3
    n2 = np.cross(v2, v3)  # Normal to the plane formed by p2, p3, p4
    # Normalize normal vectors
    n1 /= np.linalg.norm(n1)
    n2 /= np.linalg.norm(n2)
    # Calculate the cosine of the angle between n1 and n2
    cos_theta = np.dot(n1, n2)
    cos_theta = np.clip(cos_theta, -1.0, 1.0) # avoid float precision problem
    # To get the correct sign, use the scalar triple product
    sign = np.sign(np.dot(np.cross(n1, n2), v2))
    # Compute the angle in radians
    theta = np.arccos(cos_theta) * sign
    # Convert to degrees
    theta_degrees = np.degrees(theta)
    return theta_degrees

def calculate_oxygen_angle(c, o):
    """
    Calculate the azimuthal angle (theta) and polar angle (phi) 
    given points c and o in 3D space.
    
    Parameters:
    c (np.ndarray): Coordinates of point c (shape: (3,)).
    o (np.ndarray): Coordinates of point o (shape: (3,)).
    
    Returns:
    tuple: (theta, phi) angles in radians.
    """
    # Calculate the vector from c to o
    vector_co = o - c
    # Calculate the length of co
    r = np.linalg.norm(vector_co)
    # Calculate theta (azimuthal angle)
    theta1 = np.arctan2(vector_co[1], vector_co[0])
    theta1 = np.degrees(theta1)
    # Calculate phi (polar angle)
    theta2 = np.arccos(vector_co[2] / r) if r != 0 else 0  # Handle case where r = 0
    theta2 = np.degrees(theta2)
    
    return float(theta1), float(theta2)

def extract_angle_dihedrals(residues):
    # (-1CA, -1C, N, CA)  # omega
    # (-1C, N, CA, C)   # phi
    # (N, CA, C, 1N)   # psi  
    # (N, CA, C)       # tau
    # (CA, C, 1N)  
    # (C, 1N, 1CA)
    angle_dihedrals = []
    for i in range(1, len(residues) - 1):
        prev_res = residues[i - 1]
        res = residues[i]
        next_res = residues[i + 1]
        prev_C = prev_res['C'].get_coord()
        prev_CA = prev_res['CA'].get_coord()
        res_N = res['N'].get_coord()
        res_CA = res['CA'].get_coord()
        res_C = res['C'].get_coord()
        res_O = res['O'].get_coord()
        next_N = next_res['N'].get_coord()
  
        angle_dihedrals.append({
            "omega":calc_dihedral(prev_CA,prev_C,res_N,res_CA),
            "phi":calc_dihedral(prev_C,res_N,res_CA,res_C),
            "psi":calc_dihedral(res_N,res_CA,res_C,next_N),
            "dihedral_o": calc_dihedral(res_N, res_CA, res_C, res_O),
            "theta1":calc_angle(res_N, res_CA, res_C),
            "theta2":calc_angle(res_CA, res_C, next_N),
            "theta3":calc_angle(prev_C,res_N,res_CA),
            # "theta3":calc_angle(res_C, next_N, next_CA),
            "theta_o": calc_angle(res_CA, res_C, res_O),
        })
    return angle_dihedrals

# Create resid to res map mannually
def create_res_id_map(c:Chain):
    id_map = {}
    for res in c.get_residues():
        res_id = str(res.get_id()[1])
        res_icode = res.get_id()[2]
        full_id = res_id+res_icode
        id_map[full_id.strip()] = res
        if( res_id != full_id and
            res_id not in id_map.keys()):
            id_map[res_id] = res
    return id_map
import subprocess

def extract_dssp_features(structure, file_path):
    # Create temporary file
    temp_out = file_path.replace(".pdb", ".dssp")

    try:
        subprocess.run(
            ["mkdssp", "-i", file_path, "-o", temp_out],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as e:
        raise ValueError(f"DSSP failed on {file_path}:\n{e.stderr.decode()}")

    dssp = DSSP(structure[0], temp_out)

    if len(dssp) == 0:
        raise ValueError(f"DSSP returned 0 residues for {file_path}")

    # Build feature dict
    chain_id_map = {}
    dssp_features = {}
    for k in dssp.keys():
        chain_id = k[0]
        residue_id = str(k[1][1])+str(k[1][2]).strip()
        if chain_id not in chain_id_map:
            chain_id_map[chain_id] = create_res_id_map(structure[0][chain_id])
        if chain_id not in dssp_features:
            dssp_features[chain_id] = []
        dssp_features[chain_id].append({
            "res": chain_id_map[chain_id][residue_id],
            "alpha_carbon_coord": list(chain_id_map[chain_id][residue_id]["CA"].get_coord().astype(float)),
            "amino_acid": dssp[k][1],
            "secondary_structure": dssp[k][2],
            "relative_ASA": dssp[k][3],
            "NH_O_1_relidx": dssp[k][6], "NH_O_1_energy": dssp[k][7],
            "O_NH_1_relidx": dssp[k][8], "O_NH_1_energy": dssp[k][9],
            "NH_O_2_relidx": dssp[k][10], "NH_O_2_energy": dssp[k][11],
            "O_NH_2_relidx": dssp[k][12], "O_NH_2_energy": dssp[k][13],
        })
    return dssp_features

def drop_res_ojb(residue_features:list):
    for res in residue_features:
        del res["res"]
    return residue_features

def parse_by_record(record):
    msg=False
    receptor_chain_id = record.receptor_chain
    ligand_chain_id = record.ligand_chain
    structure_ids = {'pdb_id': record.pdb_id, 'receptor_chain': receptor_chain_id, 'ligand_chain': ligand_chain_id}
    
    try:
        
        # Parse file
        STRUCTURE_FOLDER = "/home2/s230112/BIB_FINAL/GNN/docking_results/"# Folder contains all the bonded complexes    
        target_filename = f"{record.pdb_id.upper()}_EMREF_1.PDB"

        # Loop through files in folder and match in case-insensitive manner
        matching_files = [
            f for f in os.listdir(STRUCTURE_FOLDER)
            if f.upper() == target_filename
        ]

        if matching_files:
            file_path = os.path.join(STRUCTURE_FOLDER, matching_files[0])
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("Complex", file_path)
        else:
            print("File Not Found")
            return [structure_ids, {"msg": "File Not Found"}]
            
        # Calculate dssp features
        features = extract_dssp_features(structure, file_path)
        print(f"{record.pdb_id} DSSP chains: {list(features.keys())}")

        # Calculate angle and dihedral features
        for chain_id in [receptor_chain_id, ligand_chain_id]:
            chain = features[chain_id]
            if chain_id not in features:
                return [structure_ids, {"msg": f"Missing chain {chain_id}"}]
            residues = [res["res"] for res in chain]
            angle_dihedrals = extract_angle_dihedrals(residues) # We calculate the angles separately for each chain and thus remove the first and last residue from each chain.
            for idx, angle_dihedral in enumerate(angle_dihedrals):
                features[chain_id][idx+1].update(angle_dihedral)
        # include pocket info.
        pocket_ids = [res_id[1:] for res_id in record.binding_site_pdb.split()]
        pocket_idx = [
            i for i, r in enumerate(features[receptor_chain_id]) 
            if str(r["res"].get_id()[1]) in pocket_ids
        ]
  
                    # Final pocket residue matching
        pocket_idx = []
        for id in pocket_ids:
            found = False
            for i, r in enumerate(features[receptor_chain_id]):
                full_id = (str(r["res"].get_id()[1]) + r["res"].get_id()[2]).strip()
                if id == full_id or id == str(r["res"].get_id()[1]).strip():
                    pocket_idx.append(i)
                    found = True
                    break
            if not found:
                print(f"{id} not found in {record.pdb_id}")

        # Return early if pocket empty
        if not pocket_idx:
            return [structure_ids, {"msg": "No pocket residues found"}]

        # Final return with clean msg
        return [structure_ids, {
            "receptor": drop_res_ojb(features[receptor_chain_id]),
            "ligand": drop_res_ojb(features[ligand_chain_id]),
            "pocket_idx": pocket_idx,
            "msg": None  # CLEAR this if no critical error
        }]
    except Exception as e:
        print(f"Failed parsing {record.pdb_id}: {e}")
        return [structure_ids,{"msg": str(e)}]


def create_data(complex_feature):
    # [1:-1]: droping first and last residue from each chain
    receptor = complex_feature[1]["receptor"][1:-1]
    ligand = complex_feature[1]["ligand"][1:-1]

    pos = [
        r["alpha_carbon_coord"] for r in receptor
    ] + [
        r["alpha_carbon_coord"] for r in ligand
    ]

    amino_acid = [
        r["amino_acid"] for r in receptor
    ] + [
        r["amino_acid"] for r in ligand
    ]

    secondary_structure = [
        r["secondary_structure"] for r in receptor
    ] + [
        r["secondary_structure"] for r in ligand
    ]
    secondary_structure = ['-' if char == 'P' else char for char in secondary_structure]
    
    numerical_features = [
        list(r.values())[3:-8:2] for r in receptor
    ] + [
        list(r.values())[3:-8:2] for r in ligand
    ]
    
    angle_features = [
        list(r.values())[-8:] for r in receptor
    ] + [
        list(r.values())[-8:] for r in ligand
    ]
    
    ligand_idx = list(range(len(receptor), len(receptor)+len(ligand)))
    pocket_idx = complex_feature[1]["pocket_idx"]
    edge_idx = [list(i) for i in itertools.product(ligand_idx, pocket_idx)] 
    pocket_mask = torch.zeros(len(receptor)+len(ligand), dtype=torch.bool)
    pocket_mask[pocket_idx] = True
    
    graph = {
        "structure_ids":complex_feature[0],
        "coors":torch.tensor(pos),
        "amino_acid":amino_acid,
        "secondary_structure":secondary_structure,
        "numerical_features":torch.tensor(numerical_features),
        "angle_features":torch.deg2rad(torch.tensor(angle_features)),
        "edge_index":torch.tensor(edge_idx).T.contiguous(),
        "ligand_mask":torch.Tensor([False]*len(receptor)+[True]*len(ligand)).bool(),
        "ligand_idx":torch.tensor(ligand_idx, dtype=torch.int),
        "pocket_mask":pocket_mask,
        "pocket_idx":torch.tensor(pocket_idx, dtype=torch.int)
    }
    return graph

def res_to_dataset(ori_data):
    print(f"Total parsed complexes: {len(ori_data)}")
    print("Messages:")
    for x in ori_data:
        if x[1]["msg"]:
            print(f"{x[0]['pdb_id']} → {x[1]['msg']}")
    # remove error data
    data = [r for r in ori_data if not r[1]["msg"]]
    #print("HELP")
    
    x_idxes = []
    for i, complex_feature in enumerate(data):
        receptor_seq = [res["amino_acid"] for res in complex_feature[1]["receptor"]]
        ligand_seq = [res["amino_acid"] for res in complex_feature[1]["ligand"]]
        print(f"{complex_feature[0]['pdb_id']} receptor: {'X' in receptor_seq}, ligand: {'X' in ligand_seq}")
        if("X" in receptor_seq or "X" in ligand_seq):
            x_idxes.append(i)
    print(f"Removing {len(x_idxes)} entries with unknown residues")
    data = [r for i,r in enumerate(data) if i not in x_idxes]
    #data = [r for r in data if len(r[1]["ligand"])>=5]
    print(data)
    result = [create_data(r) for r in data]
    return result

if __name__ == "__main__":
    complexes = pd.read_csv(BclxL_XLSX_FILE)
    complexes.drop_duplicates(subset="pdb_id", inplace=True)
    complexes.reset_index(drop=True, inplace=True)
    rows = [complexes.iloc[i] for i in range(len(complexes))]
    result = [parse_by_record(r) for r in rows]

    # Format the result
    result = res_to_dataset(result)
    # Save the result
    print(f"Built {len(result)} graphs, saving to {OUTPUT_FILE}")
    torch.save(result, OUTPUT_FILE)