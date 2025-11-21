import numpy as np
import argparse
import pickle
import json
import concepts
import time

from utils import toposort_lattice, compute_levels, compute_hierarchy, getsize


def build_parser():
    parser = argparse.ArgumentParser(description="Generate FCA lattice from class-attribute annotated dataset.")
    parser.add_argument("dataset", help="Path to the class-attribute annotated JSON dataset")
    parser.add_argument("-o", "--output", help="Path to the output lattice pickle file", default="lattice.pkl")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print lattice details")
    return parser


def prepare_and_save_lattice(args):

    with open(args.dataset, 'r') as infile:
        dataset = json.load(infile)

    all_attributes = set()
    for cls in dataset:
        all_attributes.update(dataset[cls])

    cross_table = np.zeros((len(dataset), len(all_attributes)))
    for i, cls in enumerate(dataset):
        for j, attr in enumerate(all_attributes):
            if attr in dataset[cls]:
                cross_table[i, j] = 1

    formal_context = concepts.Context(
        list(dataset.keys()), 
        all_attributes, 
        cross_table
    )
    with open(args.output, 'wb') as outfile:
        pickle.dump(formal_context, outfile)

    return dataset, all_attributes, cross_table, formal_context


def print_lattice_details(dataset, all_attributes, cross_table, formal_context):
    
    print('Number of classes:', len(dataset))
    print('Number of attributes:', len(all_attributes))
    print('Average number of attributes per class:', np.mean(np.sum(cross_table, axis=1)))

    start_time = time.time()
    lattice = formal_context.lattice
    end_time = time.time()

    print('Number of formal concepts in the lattice:', len(lattice))
    print(f'Time taken to generate the lattice: {end_time - start_time :.4f} seconds')

    sorted_concepts = toposort_lattice(lattice)
    level = compute_levels(sorted_concepts)
    hierarchy = compute_hierarchy(sorted_concepts, level)
    lattice_size = getsize(lattice)

    print('Number of hierarchy levels in the lattice:', len(hierarchy))
    print('Number of formal concepts at each hierarchy level:', [len(hierarchy[lvl]) for lvl in hierarchy])
    print(f'Size of the lattice: {lattice_size / 1e6 :.2f} MB')


if __name__ == "__main__":

    parser = build_parser()
    args = parser.parse_args()

    dataset, all_attributes, cross_table, formal_context = prepare_and_save_lattice(args)

    if args.verbose:
        print_lattice_details(dataset, all_attributes, cross_table, formal_context)