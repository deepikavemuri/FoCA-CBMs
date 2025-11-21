import numpy as np
import pandas as pd
import pickle
import json

np.random.seed(0)

DATASET = "IMAGENET100/"

with open(DATASET + "context.pkl", "rb") as pkl:
    context = pickle.load(pkl)
    lattice = context.lattice
    print(len(lattice))

with open(DATASET + "intent_levels.json", "r") as f:
    level = json.load(f)
    print(len(level))

# label = {}
# with open('labels.txt', 'r') as f:
#     for line in f.readlines():
#         class_id, index, class_name = line.strip().split()
#         label[class_id] = class_name

class_annotations = {"intent": [], "level": [], "class": [], "extent": []}
for intent in level:
    if intent == "()":
        intent_list = []
    else:
        intent_list = [x[1:-1] for x in intent[1:-1].strip(", ").split(", ")]
    try:
        extent = context.extension(intent_list)
        # print(extent)
    except Exception:
        print(intent)
        print(intent_list)
        raise RuntimeError()

    # classes = set([label[x[:9]] for x in extent])
    classes = set([x.split("_")[0] for x in extent])
    # print(intent, '\n', level[intent], '\n', extent, '\n')
    for unique_class in classes:
        class_annotations["intent"].append(intent)
        class_annotations["level"].append(level[intent])
        class_annotations["class"].append(unique_class)
        class_annotations["extent"].append(extent)

df = pd.DataFrame(class_annotations)
print(df["class"].unique())
df.to_csv(DATASET + "class_annotations.csv", index=False)
print(df)
