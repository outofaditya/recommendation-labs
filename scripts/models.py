import argparse

from export_scores import export

MODELS = [
    "Random",
    "Pop",
    "ItemKNN",
    "UserKNN",
    "BPR",
    "NeuMF",
    "LightGCN",
    "NGCF",
    "EASE",
    "SLIMElastic",
    "FISM",
]

parser = argparse.ArgumentParser()
parser.add_argument("--tuned", action="store_true")
folder = "parameters/tuned" if parser.parse_args().tuned else "parameters"

for name in MODELS:
    export(name, folder)
