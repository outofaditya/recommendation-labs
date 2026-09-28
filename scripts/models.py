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
]

for name in MODELS:
    export(name)
