from export_scores import export

MODELS = ["Random", "Pop", "ItemKNN", "UserKNN", "BPR"]

for name in MODELS:
    export(name)
