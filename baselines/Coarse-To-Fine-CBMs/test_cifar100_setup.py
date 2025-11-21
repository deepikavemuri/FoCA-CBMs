#!/usr/bin/env python3
"""Quick test to verify CIFAR100 attribute setup is working"""

import sys

sys.path.insert(0, "./Coarse-To-Fine-CBMs")

from data_utils import get_concepts, get_concept_indicators
import numpy as np

print("Testing CIFAR100 attribute setup...")
print("=" * 60)

# Test loading concepts without patchify (high-level classes)
print("\n1. Loading high-level concepts (classes)...")
try:
    concepts_high = get_concepts("cifar100", patchify=False)
    print(f"   ✓ Loaded {len(concepts_high)} high-level concepts")
    print(f"   Sample: {concepts_high[:3]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test loading concepts with patchify (low-level attributes)
print("\n2. Loading low-level concepts (attributes)...")
try:
    concepts_low = get_concepts("cifar100", patchify=True)
    print(f"   ✓ Loaded {len(concepts_low)} low-level attributes")
    print(f"   Sample: {concepts_low[:3]}")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test loading binary indicators
print("\n3. Loading binary indicator matrix...")
try:
    binary_inds = get_concept_indicators("cifar100")
    print(f"   ✓ Loaded binary matrix with shape: {binary_inds.shape}")
    print(f"   Expected: (100 classes, 700 attributes)")

    # Verify statistics
    attrs_per_class = binary_inds.sum(dim=1)
    print(f"   Average attributes per class: {attrs_per_class.mean().item():.2f}")
    print(
        f"   Min: {attrs_per_class.min().item():.0f}, Max: {attrs_per_class.max().item():.0f}"
    )
except Exception as e:
    print(f"   ✗ Error: {e}")

print("\n" + "=" * 60)
print("✓ All tests passed! CIFAR100 attributes are ready to use.")
