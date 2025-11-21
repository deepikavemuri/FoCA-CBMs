#!/usr/bin/env python3
"""
Script to create CIFAR100 attribute files from the JSON data.
Creates:
1. cifar100_attributes.txt - all unique attributes
2. cifar100_attrs_per_class_binary.npy - binary matrix mapping classes to attributes
"""

import json
import numpy as np
from pathlib import Path

# Paths
json_path = '/DATA/cifar100/cifar100_concepts.json'
output_dir = Path('/DATA/concept_sets_low/CIFAR100')
classes_file = '/DATA/concept_sets_high/CIFAR100/cifar100_classes.txt'

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Load the JSON data
with open(json_path, 'r') as f:
    cifar100_data = json.load(f)

# Load the class names in order
with open(classes_file, 'r') as f:
    class_names = [line.strip() for line in f]

print(f"Loaded {len(class_names)} classes")
print(f"Classes in JSON: {len(cifar100_data)}")

# Extract all unique attributes
all_attributes = set()
for class_name, attributes in cifar100_data.items():
    all_attributes.update(attributes)

# Sort attributes for consistency
sorted_attributes = sorted(list(all_attributes))

print(f"\nTotal unique attributes: {len(sorted_attributes)}")

# Save attributes to text file
attr_file = output_dir / 'cifar100_attributes.txt'
with open(attr_file, 'w') as f:
    for attr in sorted_attributes:
        f.write(f"{attr}\n")

print(f"Saved attributes to: {attr_file}")

# Create binary indicator matrix (classes x attributes)
# Matrix[i, j] = 1 if class i has attribute j, else 0
num_classes = len(class_names)
num_attrs = len(sorted_attributes)

# Create attribute to index mapping
attr_to_idx = {attr: idx for idx, attr in enumerate(sorted_attributes)}

# Initialize binary matrix
binary_matrix = np.zeros((num_classes, num_attrs), dtype=np.float32)

# Fill in the matrix
for class_idx, class_name in enumerate(class_names):
    if class_name in cifar100_data:
        class_attributes = cifar100_data[class_name]
        for attr in class_attributes:
            attr_idx = attr_to_idx[attr]
            binary_matrix[class_idx, attr_idx] = 1.0
    else:
        print(f"Warning: Class '{class_name}' not found in JSON data")

# Calculate statistics
attrs_per_class = binary_matrix.sum(axis=1)
classes_per_attr = binary_matrix.sum(axis=0)

print(f"\nBinary matrix shape: {binary_matrix.shape}")
print(f"Average attributes per class: {attrs_per_class.mean():.2f}")
print(f"Min attributes per class: {attrs_per_class.min():.0f}")
print(f"Max attributes per class: {attrs_per_class.max():.0f}")
print(f"Average classes per attribute: {classes_per_attr.mean():.2f}")

# Save binary matrix
matrix_file = output_dir / 'cifar100_attrs_per_class_binary.npy'
np.save(matrix_file, binary_matrix)

print(f"\nSaved binary matrix to: {matrix_file}")

# Print some sample mappings
print("\n=== Sample Class-Attribute Mappings ===")
for i in range(min(3, num_classes)):
    class_name = class_names[i]
    num_attrs_for_class = int(attrs_per_class[i])
    print(f"\n{class_name}: {num_attrs_for_class} attributes")
    # Get the attributes for this class
    attr_indices = np.where(binary_matrix[i] == 1.0)[0]
    sample_attrs = [sorted_attributes[idx] for idx in attr_indices[:5]]
    print(f"  Sample: {', '.join(sample_attrs)}")

print("\n✓ CIFAR100 attribute files created successfully!")
