"""Molecular inpainting inference wrapper around the CoCoGraph FPS models.

Adapted from `sample_scripts/sample_molecules_FPSmodel_inpaint.py` in the
original repository (https://github.com/manurubo/CoCoGraph). The core
algorithm (fragment attachment, inpainting mask, constrained denoising loop)
is unchanged; the batched multiprocessing driver used for large-scale
sampling has been replaced with a simple sequential loop suited to Ersilia's
per-request inference.
"""

import json
import math
import os

import networkx as nx
import numpy as np
import torch
from rdkit import Chem, RDLogger
from torch_geometric.data import Data

RDLogger.DisableLog("rdApp.*")

from lib_functions.adjacency_utils import components_to_graph, nx_to_rdkit
from lib_functions.config import MAX_ATOM, device
from lib_functions.data_preparation_utils import embed_edges_manuel, smiles_to_graph
from lib_functions.formula_utils import build_gt_from_formula
from lib_functions.models import (
    GINEdgeQuadrupletPredictor_MorganFP,
    GINETimePredictor_MorganFP,
)
from lib_functions.sample_utils import calculate_data_molecule_fps, filter_matrix, sample_positions

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "cocograph_data")
CHECKPOINT_DIR = os.path.join(ROOT, "..", "..", "checkpoints")

N_CANDIDATES = 100
SIGMA = 0.5  # fraction of fragment bonds used as the number of denoising swaps
MAX_STEP_RETRIES = 500  # mirrors the retry cap inside sample_step_graph_inpaint
# Retries with a fresh random fragment/attachment point. Most failures happen
# before any model inference (e.g. the composite molecule + fragment exceeds
# MAX_ATOM=70 total atoms), so retries are cheap; a generous budget matters
# most for input molecules whose size is already close to the 70-atom ceiling,
# where only a minority of the 500 fragment formulas leave enough room to fit.
MAX_CANDIDATE_RETRIES = 30


def _load_fragment_library():
    path = os.path.join(DATA_DIR, "my_fragments.txt")
    fragments = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                fragments.append(line)
    return fragments


def _load_valence_weights():
    with open(os.path.join(DATA_DIR, "valid_valences.json"), "r") as f:
        raw = json.load(f)
    valence_weights = {}
    for key, vals in raw.items():
        sym, chs = key.split("__")
        if isinstance(vals, dict):
            valence_weights[(sym, int(chs))] = {int(v): float(p) for v, p in vals.items()}
    return valence_weights


def _load_json(name):
    with open(os.path.join(DATA_DIR, name), "r") as f:
        return json.load(f)


_FRAGMENT_LIBRARY = _load_fragment_library()
_VALENCE_WEIGHTS = _load_valence_weights()
_CHARGE_SYMBOL_WEIGHTS = _load_json("charge_symbol_weights.json")
_RADICAL_SYMBOL_WEIGHTS = _load_json("radical_symbol_weights.json")


def load_models():
    """Loads the pretrained FPS diffusion and time models onto CPU in eval mode."""
    model = GINEdgeQuadrupletPredictor_MorganFP()
    time_model = GINETimePredictor_MorganFP()

    diffusion_ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "FPS_diffusion_model_epoch_1_slice_22.pth"),
        map_location="cpu",
    )
    time_ckpt = torch.load(
        os.path.join(CHECKPOINT_DIR, "FPS_time_model_epoch_2_slice_22.pth"),
        map_location="cpu",
    )
    model.load_state_dict(diffusion_ckpt["model_state_dict"])
    time_model.load_state_dict(time_ckpt["model_state_dict"])

    model.to(device).eval()
    time_model.to(device).eval()
    return model, time_model


def _get_atoms_with_hydrogens(graph):
    atoms_with_h = []
    for node, data in graph.nodes(data=True):
        if data["label"] != "H":
            for neighbor in graph.neighbors(node):
                if graph.nodes[neighbor]["label"] == "H":
                    atoms_with_h.append(node)
                    break
    return atoms_with_h


def _attach_fragment_to_molecule(original_graph, fragment_graph, rng):
    original_atoms_with_h = _get_atoms_with_hydrogens(original_graph)
    fragment_atoms_with_h = _get_atoms_with_hydrogens(fragment_graph)
    if not original_atoms_with_h:
        raise ValueError("Original molecule has no atoms with hydrogens for attachment")
    if not fragment_atoms_with_h:
        raise ValueError("Fragment has no atoms with hydrogens for attachment")

    original_attach = rng.choice(original_atoms_with_h)
    fragment_attach = rng.choice(fragment_atoms_with_h)

    composite = original_graph.copy()
    original_nodes = list(original_graph.nodes())

    fragment_node_mapping = {}
    fragment_nodes = []
    for node, data in fragment_graph.nodes(data=True):
        new_name = f"frag_{node}"
        fragment_node_mapping[node] = new_name
        composite.add_node(new_name, **data)
        fragment_nodes.append(new_name)

    for u, v, data in fragment_graph.edges(data=True):
        composite.add_edge(fragment_node_mapping[u], fragment_node_mapping[v], **data)

    h_to_remove_original = None
    for neighbor in original_graph.neighbors(original_attach):
        if original_graph.nodes[neighbor]["label"] == "H":
            h_to_remove_original = neighbor
            break

    fragment_attach_renamed = fragment_node_mapping[fragment_attach]
    h_to_remove_fragment = None
    for neighbor in list(composite.neighbors(fragment_attach_renamed)):
        if composite.nodes[neighbor]["label"] == "H" and neighbor in fragment_nodes:
            h_to_remove_fragment = neighbor
            break

    if h_to_remove_original:
        composite.remove_node(h_to_remove_original)
        original_nodes.remove(h_to_remove_original)
    if h_to_remove_fragment:
        composite.remove_node(h_to_remove_fragment)
        fragment_nodes.remove(h_to_remove_fragment)

    composite.add_edge(original_attach, fragment_attach_renamed)

    return composite, original_nodes, fragment_nodes


def _create_inpainting_mask(composite_graph, original_nodes, node_list, padded_size):
    """1 = editable, 0 = protected (bonds between original heavy atoms)."""
    mask = torch.ones((padded_size, padded_size), dtype=torch.float32)
    node_to_idx = {node: idx for idx, node in enumerate(node_list)}
    original_heavy_atoms = [
        node
        for node in original_nodes
        if node in node_to_idx and composite_graph.nodes[node]["label"] != "H"
    ]
    for node_i in original_heavy_atoms:
        idx_i = node_to_idx[node_i]
        for node_j in original_heavy_atoms:
            if node_i != node_j:
                idx_j = node_to_idx[node_j]
                mask[idx_i, idx_j] = 0
                mask[idx_j, idx_i] = 0
    return mask


def _sample_step_graph_inpaint(
    initial_graph, tensor, probs_quadrupletas_mod, all_smiles_molecule, num_swaps, contador_molecula, inpainting_mask
):
    current_graph_molecule = components_to_graph(initial_graph.nodes(data=True), tensor)
    if contador_molecula >= num_swaps:
        return tensor, None
    current_graph_molecule_copy = current_graph_molecule.copy()

    mask_des = (tensor.to(device) > 0.5).int()
    mask_haz = (tensor.to(device) < 2.5).int()
    mask_quads = (
        mask_des.unsqueeze(2).unsqueeze(3)
        * mask_des.unsqueeze(0).unsqueeze(1)
        * mask_haz.unsqueeze(1).unsqueeze(3)
        * mask_haz.unsqueeze(0).unsqueeze(2)
    )

    inpainting_mask_4d = inpainting_mask.unsqueeze(2).unsqueeze(3) * inpainting_mask.unsqueeze(0).unsqueeze(1)
    inpainting_mask_4d = inpainting_mask_4d.to(device)

    probs_quadrupletas_mod = probs_quadrupletas_mod * mask_quads * inpainting_mask_4d
    probs_quadrupletas_mod = probs_quadrupletas_mod * filter_matrix

    flat_prob_tensor = probs_quadrupletas_mod.flatten().double()

    lim_prob = 0.95
    flat_prob_tensor[flat_prob_tensor < lim_prob] = 0

    cumulative_distribution = torch.cumsum(flat_prob_tensor, dim=0)

    if cumulative_distribution[-1] == 0:
        return tensor, None

    cumulative_distribution = cumulative_distribution / cumulative_distribution[-1]

    count = 0
    tf = tensor
    modified_smiles = None
    while count < MAX_STEP_RETRIES:
        position_4d, index, error = sample_positions(cumulative_distribution, probs_quadrupletas_mod.shape)

        if error:
            count += 1
            continue

        i1, j1, i2, j2 = (
            position_4d[0].item(),
            position_4d[1].item(),
            position_4d[2].item(),
            position_4d[3].item(),
        )

        if inpainting_mask[i1, j1] == 0 or inpainting_mask[i2, j2] == 0:
            count += 1
            continue

        try:
            node_list = list(current_graph_molecule_copy.nodes())
            current_graph_molecule_copy.remove_edge(node_list[i1], node_list[j1])
            current_graph_molecule_copy.remove_edge(node_list[i2], node_list[j2])
            current_graph_molecule_copy.add_edge(node_list[i1], node_list[i2])
            current_graph_molecule_copy.add_edge(node_list[j1], node_list[j2])
        except Exception:
            count += 1
            current_graph_molecule_copy = current_graph_molecule.copy()
            continue

        if not nx.is_connected(current_graph_molecule_copy):
            count += 1
            current_graph_molecule_copy = current_graph_molecule.copy()
            continue

        try:
            mol_temporal = nx_to_rdkit(current_graph_molecule_copy, False)
            smiles_temporal = Chem.MolToSmiles(mol_temporal)

            if smiles_temporal in all_smiles_molecule:
                count += 1
                current_graph_molecule_copy = current_graph_molecule.copy()
                continue

            tf, _, _ = embed_edges_manuel(current_graph_molecule_copy, list(current_graph_molecule_copy.nodes()))
            modified_smiles = smiles_temporal
            break
        except Exception:
            count += 1
            current_graph_molecule_copy = current_graph_molecule.copy()
            continue

    return tf, modified_smiles


def _generate_one_candidate(model, time_model, original_graph, rng):
    fragment_formula = np.random.choice(_FRAGMENT_LIBRARY)
    fragment_graph = build_gt_from_formula(
        fragment_formula,
        randomize_swaps=0,
        valence_weights=_VALENCE_WEIGHTS,
        charge_symbol_weights=_CHARGE_SYMBOL_WEIGHTS,
        max_sampling_retries=100,
        allow_radicals=True,
        radical_weights=_RADICAL_SYMBOL_WEIGHTS,
    )

    composite_graph, original_nodes, _fragment_nodes = _attach_fragment_to_molecule(
        original_graph, fragment_graph, rng
    )
    composite_tensor, _, _ = embed_edges_manuel(composite_graph, list(composite_graph.nodes()))

    node_list = list(composite_graph.nodes())
    padded_size = composite_tensor.shape[0]
    inpainting_mask = _create_inpainting_mask(composite_graph, original_nodes, node_list, padded_size)

    fragment_edges = fragment_graph.number_of_edges()
    num_swaps = max(1, math.ceil(SIGMA * fragment_edges))

    all_smiles_seen = set()
    tensor = composite_tensor
    prev_time_pred = 0.5
    best_time = 0.5
    best_tensor = tensor

    for step in range(num_swaps):
        _, tensor_used, _mol, gemb, nemb, distances, edge_index, edge_attr, dosd_positions, _components, fingerprint = (
            calculate_data_molecule_fps(composite_graph, tensor, num_swaps, step)
        )

        data = Data(
            x=nemb,
            edge_index=edge_index,
            y=tensor_used,
            xA=gemb,
            edge_attr=edge_attr,
            noiselevel=torch.tensor(prev_time_pred, device=device),
            distances=torch.Tensor(distances),
            dosd_distances=dosd_positions,
            morgan_fp=fingerprint,
        ).to(device)

        with torch.no_grad():
            _, _, probs_quadrupletas_mod = model(data)
            time_pred = time_model(data).detach().cpu().item()

        if step == 0 or time_pred < best_time:
            best_time = time_pred
            best_tensor = tensor

        tensor, smiles_result = _sample_step_graph_inpaint(
            composite_graph,
            tensor.clone(),
            probs_quadrupletas_mod.detach(),
            all_smiles_seen,
            num_swaps,
            step,
            inpainting_mask,
        )
        if smiles_result is not None:
            all_smiles_seen.add(smiles_result)

        prev_time_pred = time_pred

    g_final = components_to_graph(composite_graph.nodes(data=True), best_tensor)
    mol_final = nx_to_rdkit(g_final, False)
    return Chem.MolToSmiles(mol_final)


def generate_candidates(model, time_model, smiles, n_candidates=N_CANDIDATES):
    """Generates `n_candidates` small-fragment-inpainted analogues of `smiles`.

    Returns a list of `n_candidates` SMILES strings, using empty strings for
    candidates that could not be generated (invalid input molecule, or a
    fragment/attachment attempt that repeatedly failed).
    """
    original_graph = smiles_to_graph(smiles)
    if original_graph is None or original_graph.number_of_nodes() > MAX_ATOM:
        return [""] * n_candidates

    results = []
    for _ in range(n_candidates):
        smiles_out = ""
        for _attempt in range(MAX_CANDIDATE_RETRIES):
            try:
                rng = np.random.default_rng()
                smiles_out = _generate_one_candidate(model, time_model, original_graph, rng)
                break
            except Exception:
                continue
        results.append(smiles_out)
    return results
