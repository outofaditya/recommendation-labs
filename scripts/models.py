from export_scores import export

MODELS = ["Random", "Pop", "ItemKNN"]

for name in MODELS:
    export(name)
