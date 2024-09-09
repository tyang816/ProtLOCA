import json
import argparse
import torch
import torch_cluster
import math
import os
import sys
sys.path.append(os.getcwd())
import random
import pandas as pd
import numpy as np
import scipy.spatial as spa
import networkx as nx
import torch.nn.functional as F
from tqdm import tqdm
from Bio import PDB
from Bio.SeqUtils import seq1
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
from src.utils.foldseek_utils import convert_fasta_to_dic

ss8_vocab = [
    'E', 'L', 'I', 'T', 'H', 'B', 'G', 'S'
    ]
foldseek_vocab = [
    'P', 'Y', 'N', 'W', 'R', 'Q', 'H', 'G', 'D', 'L', 
    'V', 'T', 'M', 'F', 'S', 'A', 'E', 'I', 'K', 'C',
    ]
ss3_vocab = ['H', 'E', 'C']
ss8_to_ss3_dict = {
    "H": "H", "G": "H", "E": "E",
    "B": "E", "I": "C", "T": "C",
    "S": "C", "L": "C", "-": "C",
    "P": "C",'C': "C"
}
aa_vocab = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
    'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V',
    'X', 'U', 'O', 'B', 'Z', 'J'
    ]
# 20x8 = 160
foldseek_ss8_vocab = [
    'PE', 'PL', 'PI', 'PT', 'PH', 'PB', 'PG', 'PS',
    'YE', 'YL', 'YI', 'YT', 'YH', 'YB', 'YG', 'YS',
    'NE', 'NL', 'NI', 'NT', 'NH', 'NB', 'NG', 'NS',
    'WE', 'WL', 'WI', 'WT', 'WH', 'WB', 'WG', 'WS',
    'RE', 'RL', 'RI', 'RT', 'RH', 'RB', 'RG', 'RS',
    'QE', 'QL', 'QI', 'QT', 'QH', 'QB', 'QG', 'QS',
    'HE', 'HL', 'HI', 'HT', 'HH', 'HB', 'HG', 'HS',
    'GE', 'GL', 'GI', 'GT', 'GH', 'GB', 'GG', 'GS',
    'DE', 'DL', 'DI', 'DT', 'DH', 'DB', 'DG', 'DS',
    'LE', 'LL', 'LI', 'LT', 'LH', 'LB', 'LG', 'LS',
    'VE', 'VL', 'VI', 'VT', 'VH', 'VB', 'VG', 'VS',
    'TE', 'TL', 'TI', 'TT', 'TH', 'TB', 'TG', 'TS',
    'ME', 'ML', 'MI', 'MT', 'MH', 'MB', 'MG', 'MS',
    'FE', 'FL', 'FI', 'FT', 'FH', 'FB', 'FG', 'FS',
    'SE', 'SL', 'SI', 'ST', 'SH', 'SB', 'SG', 'SS',
    'AE', 'AL', 'AI', 'AT', 'AH', 'AB', 'AG', 'AS',
    'EE', 'EL', 'EI', 'ET', 'EH', 'EB', 'EG', 'ES',
    'IE', 'IL', 'II', 'IT', 'IH', 'IB', 'IG', 'IS',
    'KE', 'KL', 'KI', 'KT', 'KH', 'KB', 'KG', 'KS',
    'CE', 'CL', 'CI', 'CT', 'CH', 'CB', 'CG', 'CS',
]
aa_foldseek_vocab = [aa+f for aa in aa_vocab for f in foldseek_vocab]
aa_ss8_vocab = [aa+f for aa in aa_vocab for f in ss8_vocab]
aa_foldseek_ss8_vocab = [aa+f for aa in aa_vocab for f in foldseek_ss8_vocab]



def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True))
        )

def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

def _orientations(X_ca):
    forward = _normalize(X_ca[1:] - X_ca[:-1])
    backward = _normalize(X_ca[:-1] - X_ca[1:])
    forward = F.pad(forward, [0, 0, 0, 1])
    backward = F.pad(backward, [0, 0, 1, 0])
    return torch.cat([forward.unsqueeze(-2), backward.unsqueeze(-2)], -2)

def _sidechains(X):
    n, origin, c = X[:, 0], X[:, 1], X[:, 2]
    c, n = _normalize(c - origin), _normalize(n - origin)
    bisector = _normalize(c + n)
    perp = _normalize(torch.cross(c, n))
    vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
    return vec 

def _positional_embeddings(edge_index, 
                            num_embeddings=16,
                            period_range=[2, 1000]):
    # From https://github.com/jingraham/neurips19-graph-protein-design
    d = edge_index[0] - edge_index[1]
    
    frequency = torch.exp(
        torch.arange(0, num_embeddings, 2, dtype=torch.float32)
        * -(np.log(10000.0) / num_embeddings)
    )
    angles = d.unsqueeze(-1) * frequency
    E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
    return E


def choose_pdb(path):
    with open(os.path.join(path, "ranking_debug.json"), "r") as f:
        score = json.load(f)
    pdb = score["order"][0]
    pdb_path = os.path.join(path, "unrelaxed_" + pdb + ".pdb")
    return pdb_path

def to_networkx(graph_data, node_attrs=["node_s", "node_v"], edge_attrs=["edge_s", "edge_v"]):
    G = nx.Graph()
    G.add_nodes_from(range(graph_data.num_nodes))
    edge_list = graph_data.edge_index.t().tolist()
    
    values = {}
    for key, value in graph_data(*(node_attrs + edge_attrs)):
        if torch.is_tensor(value):
            value = value if value.dim() <= 1 else value.squeeze(-1)
            values[key] = value.tolist()
        else:
            values[key] = value
    
    for i, (u, v) in enumerate(edge_list):
        G.add_edge(u, v)
        for key in edge_attrs:
            G[u][v][key] = values[key][i]
    
    for key in node_attrs:
        for i, feat_dict in G.nodes(data=True):
            feat_dict.update({key: values[key][i]})
    return G

    

def generate_graph(pdb_file, node_level, node_s_type, max_distance=10, foldseek_fasta_file=None, foldseek_fasta_multi_chain=False):
    """
    generate graph data from pdb file
    
    params:
        pdb_file: pdb file path
        node_level: residue or secondary_structure
        node_s_type: ss3, ss8, foldseek, foldseek_ss8, aa
        max_distance: cut off
        foldseek_fasta_file: foldseek fasta file path
        foldseek_fasta_multi_chain: pdb multi chain for foldseek fasta
    
    return:
        graph data
    
    """
    pdb_parser = PDB.PDBParser(QUIET=True)
    structure = pdb_parser.get_structure("protein", pdb_file)
    model = structure[0]

    # extract amino acid sequence
    seq = []
    # extract amino acid coordinates
    aa_coords = {"N": [], "CA": [], "C": [], "O": []}
    coord_anchors = ["N", "CA", "C", "O"]
    
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.get_id()[0] == " ":
                    seq.append(residue.get_resname())
                    for atom_name in aa_coords.keys():
                        atom = residue[atom_name]
                        aa_coords[atom_name].append(atom.get_coord().tolist())
    aa_seq = "".join([seq1(aa) for aa in seq])
    
    if node_level == "secondary_structure":
        dssp = PDB.DSSP(model, pdb_file)
        # for each secondary structure, extract the coordinates of all amino acids
        ss_coords_dict = {"N": {}, "CA": {}, "C": {}, "O": {}}
        ss_coords_list = {"N": [], "CA": [], "C": [], "O": []}
        previous_ss = None
        ss8_seq_list = []
        j = -1
        # for each amino acid, find its secondary structure
        for i, dssp_res in enumerate(dssp):
            ss_token = dssp_res[2].replace('-', 'L')
            if node_s_type == "ss3":
                ss_token = ss8_to_ss3_dict[ss_token]
            if ss_token != previous_ss:
                j += 1
                ss8_seq_list.append(ss_token)
                previous_ss = ss_token
            for anchor in coord_anchors:
                if j not in ss_coords_dict[anchor].keys():
                    ss_coords_dict[anchor][j] = []
                ss_coords_dict[anchor][j].append(aa_coords[anchor][i])
        # calculate the mean coordinates of each secondary structure
        for anchor in coord_anchors:
            for k, v in ss_coords_dict[anchor].items():
                ss_coords_dict[anchor][k] = [sum(e)/len(e) for e in zip(*v)]
                ss_coords_list[anchor].append(ss_coords_dict[anchor][k])
        coords = list(zip(ss_coords_list['N'], ss_coords_list['CA'], ss_coords_list['C'], ss_coords_list['O']))
        coords = torch.tensor(coords)
        # mask out the missing coordinates
        mask = torch.isfinite(coords.sum(dim=(1,2)))
        coords[~mask] = np.inf
        ca_coords = coords[:, 0]
    
        # one-hot for secondary structure
        ss_seq = ''.join(ss8_seq_list)
        integer_encoded = [ss3_vocab.index(char) for char in ss_seq]
        ss_node = F.one_hot(torch.tensor(integer_encoded), num_classes=len(ss8_vocab))
        node_s = ss_node

        edge_index = torch_cluster.knn_graph(ca_coords, k=3)
        distances = spa.distance_matrix(ca_coords, ca_coords)
        edge_distances = distances[edge_index[0].numpy(), edge_index[1].numpy()]

        mask = edge_distances < max_distance
        edge_index = edge_index[:, mask]
        
    elif node_level == "residue":
        # aa means amino acid
        coords = list(zip(aa_coords['N'], aa_coords['CA'], aa_coords['C'], aa_coords['O']))
        coords = torch.tensor(coords)
        # mask out the missing coordinates
        mask = torch.isfinite(coords.sum(dim=(1,2)))
        coords[~mask] = np.inf
        ca_coords = coords[:, 1]
        
        if node_s_type == "ss3" or node_s_type == "ss8":
            dssp = PDB.DSSP(model, pdb_file)
            ss8_seq_list = []
            for i, dssp_res in enumerate(dssp):
                ss8_seq_list.append(dssp_res[2])
            assert len(ss8_seq_list) == len(ca_coords), f"num of sec_structure not equal to ca_coords {len(ss8_seq_list)} != {len(ca_coords)}"
            
            # replace '-' with 'L'
            ss8_seq = ''.join(ss8_seq_list)
            ss8_seq = ss8_seq.replace('-', 'L')
            ss8_seq_list = list(ss8_seq)
            
            # one-hot for secondary structure
            if node_s_type == "ss3":
                node_s = F.one_hot(
                    torch.tensor([ss3_vocab.index(ss8_to_ss3_dict[char]) for char in ss8_seq_list]), 
                    num_classes=len(ss3_vocab)
                )
            elif node_s_type == "ss8":
                node_s = F.one_hot(
                    torch.tensor([ss8_vocab.index(char) for char in ss8_seq_list]),
                    num_classes=len(ss8_vocab)
                )
                    
        elif node_s_type == "foldseek":
            if foldseek_fasta_file is not None:
                # we add foldseek feature to node_s, no ss feature
                foldseek_dic = convert_fasta_to_dic(foldseek_fasta_file, foldseek_fasta_multi_chain)
                foldseek_seq = foldseek_dic[pdb_file.split('/')[-1]]
                assert len(foldseek_seq) == len(ca_coords), f"num of foldseek not equal to ca_coords, {len(foldseek_seq)} != {len(ca_coords)}"
                node_s = F.one_hot(
                    torch.tensor([foldseek_vocab.index(char) for char in foldseek_seq]),
                    num_classes=len(foldseek_vocab)
                )
            else:
                node_s = torch.zeros(len(ca_coords), len(foldseek_vocab))
        
        elif node_s_type == "foldseek_ss8":
            dssp = PDB.DSSP(model, pdb_file)
            # we add ss feature to node_s, no aa feature
            ss8_seq_list = []
            for i, dssp_res in enumerate(dssp):
                ss8_seq_list.append(dssp_res[2])
            assert len(ss8_seq_list) == len(ca_coords), f"num of sec_structure not equal to ca_coords {len(ss8_seq_list)} != {len(ca_coords)}"
            
            # replace '-' with 'L'
            ss8_seq = ''.join(ss8_seq_list)
            ss8_seq = ss8_seq.replace('-', 'L')
            
            assert foldseek_fasta_file is not None, "foldseek fasta file is required"
            foldseek_dic = convert_fasta_to_dic(foldseek_fasta_file, foldseek_fasta_multi_chain)
            foldseek_seq = foldseek_dic[pdb_file.split('/')[-1]]
            assert len(foldseek_seq) == len(ca_coords), f"num of foldseek not equal to ca_coords, {len(foldseek_seq)} != {len(ca_coords)}"
            node_s = F.one_hot(
                torch.tensor([foldseek_ss8_vocab.index(f"{foldseek_seq[i]}{ss8_seq[i]}") for i in range(len(ca_coords))]), 
                num_classes=len(foldseek_ss8_vocab)
            )
        
        elif node_s_type == "aa_foldseek":
            if foldseek_fasta_file is not None:
                # we add foldseek feature to node_s, no ss feature
                foldseek_dic = convert_fasta_to_dic(foldseek_fasta_file, foldseek_fasta_multi_chain)
                foldseek_seq = foldseek_dic[pdb_file.split('/')[-1]]
                assert len(foldseek_seq) == len(ca_coords), f"num of foldseek not equal to ca_coords, {len(foldseek_seq)} != {len(ca_coords)}"
                node_s = F.one_hot(
                    torch.tensor([aa_foldseek_vocab.index(f"{aa_seq[i]}{foldseek_seq[i]}") for i in range(len(ca_coords))]),
                    num_classes=len(aa_foldseek_vocab)
                )
            else:
                node_s = torch.zeros(len(ca_coords), len(aa_foldseek_vocab))
        
        elif node_s_type == "aa_ss8":
            dssp = PDB.DSSP(model, pdb_file)
            # we add ss feature to node_s, no aa feature
            ss8_seq_list = []
            for i, dssp_res in enumerate(dssp):
                ss8_seq_list.append(dssp_res[2])
            assert len(ss8_seq_list) == len(ca_coords), f"num of sec_structure not equal to ca_coords {len(ss8_seq_list)} != {len(ca_coords)}"
            
            # replace '-' with 'L'
            ss8_seq = ''.join(ss8_seq_list)
            ss8_seq = ss8_seq.replace('-', 'L')
            node_s = F.one_hot(
                torch.tensor([aa_ss8_vocab.index(f"{aa_seq[i]}{ss8_seq[i]}") for i in range(len(ca_coords))]),
                num_classes=len(aa_ss8_vocab)
            )
        
        elif node_s_type == "aa_foldseek_ss8":
            dssp = PDB.DSSP(model, pdb_file)
            # we add ss feature to node_s, no aa feature
            ss8_seq_list = []
            for i, dssp_res in enumerate(dssp):
                ss8_seq_list.append(dssp_res[2])
            assert len(ss8_seq_list) == len(ca_coords), f"num of sec_structure not equal to ca_coords {len(ss8_seq_list)} != {len(ca_coords)}"
            
            # replace '-' with 'L'
            ss8_seq = ''.join(ss8_seq_list)
            ss8_seq = ss8_seq.replace('-', 'L')
            
            assert foldseek_fasta_file is not None, "foldseek fasta file is required"
            foldseek_dic = convert_fasta_to_dic(foldseek_fasta_file, foldseek_fasta_multi_chain)
            foldseek_seq = foldseek_dic[pdb_file.split('/')[-1]]
            assert len(foldseek_seq) == len(ca_coords), f"num of foldseek not equal to ca_coords, {len(foldseek_seq)} != {len(ca_coords)}"
            node_s = F.one_hot(
                torch.tensor([aa_foldseek_ss8_vocab.index(f"{aa_seq[i]}{foldseek_seq[i]}{ss8_seq[i]}") for i in range(len(ca_coords))]),
                num_classes=len(aa_foldseek_ss8_vocab)
            )
        
        elif node_s_type == "aa":
            node_s = F.one_hot(
                torch.tensor([aa_vocab.index(aa) for aa in aa_seq]),
                num_classes=len(aa_vocab)
            )
        
        # build graph and max_distance
        distances = spa.distance_matrix(ca_coords, ca_coords)
        edge_index = torch.tensor(np.array(np.where(distances < max_distance)))
        # remove loop
        mask = edge_index[0] != edge_index[1]
        edge_index = edge_index[:, mask]
    
    # node features
    orientations = _orientations(ca_coords)
    sidechains = _sidechains(coords)
    node_v = torch.cat([orientations, sidechains.unsqueeze(-2)], dim=-2)
    
    # edge features
    pos_embeddings = _positional_embeddings(edge_index)
    E_vectors = ca_coords[edge_index[0]] - ca_coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_count=16)
    edge_s = torch.cat([rbf, pos_embeddings], dim=-1)
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    
    # node_v: [node_num, 3, 3]
    # edge_index: [2, edge_num]
    # edge_s: [edge_num, 16+16]
    # edge_v: [edge_num, 1, 3]
    node_s, node_v, edge_s, edge_v = map(torch.nan_to_num, (node_s, node_v, edge_s, edge_v))
    data = Data(
        node_s=node_s, node_v=node_v, 
        edge_index=edge_index, 
        edge_s=edge_s, edge_v=edge_v,
        distances=distances,
        aa_seq=aa_seq,
        ca_coords=ca_coords
    )
    
    return data

def generate_graph_report(args):
    graphs = os.listdir(args.graph_out_dir)
    aa_len, ss_len, node_num, egde_num = [], [], [], []
    for g in tqdm(graphs, desc="graph report"):
        graph_data = torch.load(os.path.join(args.graph_out_dir, g))
        aa_len.append(len(graph_data.aa_seq))
        ss_len.append(len(graph_data.ss_seq))
        node_num.append(graph_data.num_nodes)
        egde_num.append(graph_data.num_edges)
    print(f"aa_len: {np.mean(aa_len)}; max: {np.max(aa_len)}; min: {np.min(aa_len)}")
    print(f"ss_len: {np.mean(ss_len)}; max: {np.max(ss_len)}; min: {np.min(ss_len)}")
    print(f"node_num: {np.mean(node_num)}; max: {np.max(node_num)}; min: {np.min(node_num)}")
    print(f"egde_num: {np.mean(egde_num)}; max: {np.max(egde_num)}; min: {np.min(egde_num)}")
    
    subgraphs = os.listdir(args.subgraph_out_dir)
    node_num, egde_num = [], []
    for g in tqdm(subgraphs, desc="subgraph report"):
        sg = os.listdir(os.path.join(args.subgraph_out_dir, g))
        for s in sg:
            subgraph_data = torch.load(os.path.join(args.subgraph_out_dir, g, s))
            node_num.append(subgraph_data.num_nodes)
            egde_num.append(subgraph_data.num_edges)
    print(f"subgraph 【node_num】 mean: {np.mean(node_num)}; max: {np.max(node_num)}; min: {np.min(node_num)}")
    print(f"subgraph 【egde_num】 mean: {np.mean(egde_num)}; max: {np.max(egde_num)}; min: {np.min(egde_num)}")
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--pdb_file", default=None, type=str, help="pdb file")
    parser.add_argument("--node_level", choices=["residue", "secondary_structure"], default="residue", type=str, help="node level")
    parser.add_argument("--node_s_type", choices=["ss3", "ss8", "foldseek", "foldseek_ss8", "aa", "aa_foldseek", "aa_ss8", "aa_foldseek_ss8"], default="ss3", type=str, help="node_s type")
    parser.add_argument("--foldseek_fasta_file", default=None, type=str, help="foldseek fasta file")
    parser.add_argument("--foldseek_fasta_multi_chain", action="store_true", help="pdb multi chain for foldseek fasta")
    parser.add_argument("--knn", default=None, type=int, help="knn")
    parser.add_argument("--max_distance", default=10, type=int, help="cut off")
    parser.add_argument("--pdb_dir", default=None, type=str, help="pdb dir")
    parser.add_argument("--num_workers", default=16, type=int, help="num workers for multiprocessing")
    parser.add_argument("--graph_out_dir", required=True, type=str, help="graph out dir")
    parser.add_argument("--error_file", default=None, type=str, help="csv error file")
    args = parser.parse_args()
    
    
    os.makedirs(args.graph_out_dir, exist_ok=True)
    
    
    if args.pdb_dir is not None:
        def process_protein(p, args):
            pdb_file = os.path.join(args.pdb_dir, p)
            name = p.split('.')[0]
            if os.path.exists(os.path.join(args.graph_out_dir, name + ".pt")):
                return None
            try:
                graph_data = generate_graph(
                    pdb_file, args.node_level, args.node_s_type, args.max_distance, 
                    args.foldseek_fasta_file, args.foldseek_fasta_multi_chain
                    )
                torch.save(graph_data, os.path.join(args.graph_out_dir, name + ".pt"))
                return None
            except Exception as e:
                return p, str(e)

        root_dir = "/".join(args.pdb_dir.split("/")[:-1])
        proteins = sorted(os.listdir(args.pdb_dir))
        error_proteins = []
        error_messages = []

        with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_protein, p, args): p for p in proteins}

            for future in tqdm(as_completed(futures), total=len(proteins), desc="Processing Proteins"):
                result = future.result()
                if result is not None:
                    error_proteins.append(result[0])
                    error_messages.append(result[1])
                    print(f"> Error processing {result[0]}: {result[1]}")

        if args.error_file is not None:
            error_dict = {"protein": error_proteins, "error": error_messages}
            error_dir = os.path.dirname(args.error_file)
            os.makedirs(error_dir, exist_ok=True)
            pd.DataFrame(error_dict).to_csv(args.error_file, index=False)
            print(f">>> Total error proteins: {len(error_proteins)}")
        else:
            print(f">>> Error proteins: {error_proteins}")
            print(f">>> Success proteins: {len(proteins) - len(error_proteins)}")
    
    if args.pdb_file is not None:
        graph_data = generate_graph(
            args.pdb_file, args.node_level, args.node_s_type, args.max_distance, 
            args.foldseek_fasta_file, args.foldseek_fasta_multi_chain
            )
        torch.save(
            graph_data, os.path.join(args.graph_out_dir, args.pdb_file.split('/')[-1].split('.')[0] + ".pt"))