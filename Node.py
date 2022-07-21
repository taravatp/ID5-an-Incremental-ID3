import numpy as np
import math


class Node:

    def __init__(self, feature_index=None, edge_value=None, children=None, parent=None, visited_samples=None):
        if visited_samples is None:
            visited_samples = []
        self.feature_index = feature_index  # which feature this node represent
        self.edge_value = edge_value  # represents feature value
        self.parent = parent    # parent node
        self.children = children    # a dictionary of child nodes
        self.visited_samples = visited_samples  # samples that have reached this node so far
        self.omit = False   # this will be used during pruning

    def get_entropy(self):
        samples = np.array(self.visited_samples)
        labels = samples[:, -1]
        total = len(samples)
        labels, counts = np.unique(labels, return_counts=True)
        entropy = 0
        for count in counts:
            entropy += -(count / total) * math.log10(count / total)
        return entropy

    def get_label(self):
        visited = np.array(self.visited_samples)
        values, counts = np.unique(visited[:, -1], return_counts=True)
        label = values[np.argmax(counts)]
        return label

    def __repr__(self, level=0):
        if self.feature_index is None:
            show = "label: " + str(self.get_label())
        else:
            show = 'feature:'+str(self.feature_index)

        ret = "\t" * level + repr(show) + "\n"
        if self.children is not None:
            for child in self.children:
                ret += self.children[child].__repr__(level + 1)
        return ret
