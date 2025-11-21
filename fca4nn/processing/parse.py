import numpy as np
import json
import pickle
import concepts

np.random.seed(0)

with open("inet100_instanceconcept.json", "r") as f:
    data = json.load(f)

labels = {}
max_len = 0
with open("labels.txt", "r") as f:
    for line in f.readlines():
        class_id, index, class_name = line.strip().split()
        labels[class_id] = class_name
        max_len = max(max_len, len(class_name))
print(max_len)

# pprint(list(data.keys()))
print(len(data.keys()))
print([len(data[key]) for key in data.keys()])

all_attributes = []
attribute_counts = {}
max_attributes, min_attributes = 0, 1e9
for key in data.keys():
    for subkey in data[key].keys():
        all_attributes.extend(data[key][subkey])
        max_attributes = max(max_attributes, len(data[key][subkey]))
        min_attributes = min(min_attributes, len(data[key][subkey]))
        for attr in data[key][subkey]:
            if attr in attribute_counts:
                attribute_counts[attr] += 1
            else:
                attribute_counts[attr] = 1
all_attributes = set(all_attributes)
# pprint(all_attributes)
# pprint({k: v for k, v in sorted(attribute_counts.items(), key=lambda item: item[1], reverse=True)[:20]}, sort_dicts=False)
print(len(all_attributes), min_attributes, max_attributes)
print("-" * 50)

NUM_CLASSES = 10
NUM_SAMPLES = 10

context_dict = {}
for key in np.random.choice(list(data.keys()), NUM_CLASSES):
    class_attributes = all_attributes.copy()
    for subkey in np.random.choice(list(data[key].keys()), NUM_SAMPLES):
        image_attributes = set(data[key][subkey])
        class_attributes &= image_attributes
        context_dict[subkey] = list(image_attributes)
    if len(class_attributes):
        # print(f'{key} | {labels[key]:<15} | {class_attributes}')
        # context_dict[labels[key]] = list(class_attributes)
        pass

for key in ["n01644373", "n01914609"]:
    class_counts = {}
    for subkey in data[key].keys():
        for attr in data[key][subkey]:
            if attr in class_counts:
                class_counts[attr] += 1
            else:
                class_counts[attr] = 1
    class_counts = {
        k: v
        for k, v in sorted(
            class_counts.items(), key=lambda item: item[1], reverse=True
        )[19::-1]
    }
    # plt.barh(list(class_counts.keys()), list(class_counts.values()))
    # plt.title(labels[key])
    # plt.show()

context_objects = context_dict.keys()
context_attributes = []
for value in context_dict.values():
    context_attributes.extend(value)
context_attributes = list(set(context_attributes))
context_matrix = [
    [(attr in context_dict[obj]) for attr in context_attributes]
    for obj in context_objects
]
print(np.array(context_matrix).shape)

context = concepts.Context(context_objects, context_attributes, context_matrix)
# print(context.extension(['amphibian']))
count = 0
for extent, intent in context.lattice:
    if len(extent) > 1 and len(set([x[:9] for x in extent])) > 1:
        # print(extent, intent)
        print(
            f"{len(extent)} images, {len(set([x[:9] for x in extent]))} classes, {intent}"
        )
        count += 1
print(len(context.lattice))
print(count)
with open("lattice.pkl", "wb") as pkl:
    pickle.dump(context, pkl)
