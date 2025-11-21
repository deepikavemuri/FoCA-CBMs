import sys
import networkx as nx
from types import ModuleType, FunctionType
from gc import get_referents


def toposort_lattice(lattice):
    G = nx.DiGraph()
    for concept in lattice:
        G.add_node(concept)
        for upper in concept.upper_neighbors:
            G.add_edge(concept, upper) 
    sorted_concepts = list(nx.topological_sort(G))
    return sorted_concepts


def compute_levels(concepts):
    level = {}
    for concept in concepts:
        extent, intent = concept
        if str(extent) not in level:
            level[str(extent)] = 0
        for upper in concept.upper_neighbors:
            upper_extent, upper_intent = upper
            if str(upper_extent) not in level:
                level[str(upper_extent)] = level[str(extent)] + 1
            else:
                level[str(upper_extent)] = max(level[str(upper_extent)], level[str(extent)] + 1)
    return level


def compute_hierarchy(concepts, level):
    hierarchy = {}
    for concept in concepts:
        extent, intent = concept
        lvl = level[str(extent)]
        if lvl in hierarchy:
            hierarchy[lvl].append(concept)
        else:
            hierarchy[lvl] = [concept]
    return hierarchy


def getsize(obj):
    """ https://stackoverflow.com/a/30316760 """
    BLACKLIST = type, ModuleType, FunctionType
    if isinstance(obj, BLACKLIST):
        raise TypeError('getsize() does not take argument of type: '+ str(type(obj)))
    seen_ids = set()
    size = 0
    objects = [obj]
    while objects:
        need_referents = []
        for obj in objects:
            if not isinstance(obj, BLACKLIST) and id(obj) not in seen_ids:
                seen_ids.add(id(obj))
                size += sys.getsizeof(obj)
                need_referents.append(obj)
        objects = get_referents(*need_referents)
    return size