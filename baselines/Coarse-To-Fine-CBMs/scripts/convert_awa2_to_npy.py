#!/usr/bin/env python3
import numpy as np
import os

in_path = './DATA/AWA2/Animals_with_Attributes2/predicate-matrix-binary.txt'
out_path = './DATA/concept_sets_low/AwA2/awa2_attrs_per_class_binary_85.npy'

with open(in_path, 'r') as f:
    lines = [l.strip() for l in f if l.strip()]
rows = [list(map(int, l.split())) for l in lines]
arr = np.array(rows, dtype=np.uint8)
print('loaded shape', arr.shape)
# transpose if file is attributes x classes -> make classes x attributes
if arr.shape[0] > arr.shape[1]:
    arr = arr.T
print('final shape (classes x attrs)', arr.shape)
os.makedirs(os.path.dirname(out_path), exist_ok=True)
np.save(out_path, arr)
print('saved', out_path)
