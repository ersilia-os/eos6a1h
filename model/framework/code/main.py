# imports
import csv
import os
import sys

from ersilia_pack_utils.core import read_smiles

from cocograph_inpaint import N_CANDIDATES, generate_candidates, load_models

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))

# load the pretrained diffusion and time models once
model, time_model = load_models()


# my model
def my_model(smiles_list):
    return [generate_candidates(model, time_model, smi, N_CANDIDATES) for smi in smiles_list]


# read SMILES from .csv file, assuming one column with header
_, smiles_list = read_smiles(input_file)

# run model
outputs = my_model(smiles_list)

# check input and output have the same length
assert len(smiles_list) == len(outputs)

# write output in a .csv file
header = [f"smi_{str(i).zfill(2)}" for i in range(N_CANDIDATES)]
with open(output_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    for row in outputs:
        writer.writerow(row)
