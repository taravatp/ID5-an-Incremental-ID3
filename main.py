import math
import numpy as np
import pandas as pd
from Node import Node
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score


def calculate_samples_entropy(current_samples):
    current_samples = np.array(current_samples)
    current_labels = current_samples[:, -1]
    total = len(current_labels)
    labels, counts = np.unique(current_labels, return_counts=True)
    entropy = 0
    for count in counts:
        entropy += -(count / total) * math.log10(count / total)
    return entropy


def calculate_information_gain_branching(current_samples, parent_entropy):
    current_samples = np.array(current_samples)
    features_info_gain = {}
    for feature in features:  # test all of the features
        if used_features[feature] == 1:  # you can use this feature
            child_entropy = 0
            values, counts = np.unique(current_samples[:, feature], return_counts=True)
            for feature_value in values:  # test all the feature values of each feature
                labels = current_samples[current_samples[:, feature] == feature_value][:, -1]
                child_entropy += -(len(labels) / len(current_samples)) * math.log10(len(labels) / len(current_samples))
            features_info_gain[
                feature] = child_entropy - parent_entropy
    return features_info_gain


def get_feature_values(current_samples, feature_index):
    current_samples = np.array(current_samples)
    feature_values = np.unique(current_samples[:, feature_index])
    return feature_values


def insert_root(current_samples):
    global used_features

    parent_entropy = calculate_samples_entropy(current_samples)
    features_info_gain = calculate_information_gain_branching(current_samples, parent_entropy)
    feature_index = max(features_info_gain, key=features_info_gain.get)
    used_features[feature_index] = 0  # you can no longer use this feature!

    root = Node(feature_index=feature_index)
    children = {}
    for feature_value in get_feature_values(current_samples, feature_index):
        child = Node(edge_value=feature_value, parent=root)
        children[feature_value] = child
    root.children = children
    return root


def add_data(node, data):
    feature_index = node.feature_index
    feature_value = data[feature_index]
    node.visited_samples.append(data)  # update that node
    extend_node = None

    if feature_value in node.children:
        if node.children[feature_value].feature_index is not None:  # we are not in a leaf node
            return add_data(node.children[feature_value], data)
        else:  # we are in a leaf node
            node.children[feature_value].visited_samples.append(data)
            entropy = node.children[feature_value].get_entropy()
            if entropy != 0:  # a new node should be added - extending the tree
                extend_node = node.children[feature_value]  # in mishe parente nodi ke gharare ziresh node ezafe beshe
    else:  # we should add a branch to our current node
        new_node = Node(edge_value=feature_value, parent=node, visited_samples=[data])
        node.children[feature_value] = new_node
    return node, extend_node


def pull_up(root, current_samples):
    # step1 - first we are going to find the best child
    current_samples = np.array(current_samples)
    best_child = None
    min_entropy = root.get_entropy()
    for child in root.children:
        child_node = root.children[child]
        if child_node.feature_index is not None:  # if this is not a leaf node! - we just want the test nodes!
            if child_node.get_entropy() < min_entropy:
                min_entropy = child_node.get_entropy()
                best_child = child_node

    if best_child is None:
        return

    #  step1 - then we are going to create multiple trees
    step1_roots = []
    for child in root.children:  # we are creating one tree in each iteration
        new_root_1 = Node(feature_index=root.feature_index, children={},
                          visited_samples=root.children[child].visited_samples)
        new_root_2 = Node(feature_index=best_child.feature_index, edge_value=child, parent=new_root_1,
                          visited_samples=root.children[child].visited_samples)
        new_root_1.children[child] = new_root_2

        children = {}
        for feature_value in get_feature_values(new_root_2.visited_samples, best_child.feature_index):
            visited = np.array(new_root_2.visited_samples).reshape(-1, 11)  # age bardaram chi mishe??
            visited = visited[visited[:, best_child.feature_index] == feature_value]
            new_child = Node(feature_index=None, edge_value=feature_value, parent=new_root_2,
                             visited_samples=list(visited))  # what should be the index of child?
            children[feature_value] = new_child

        new_root_2.children = children
        step1_roots.append(new_root_1)

    # step2
    step2_roots = []
    for r in step1_roots:
        new_root_1 = Node(feature_index=best_child.feature_index, visited_samples=r.visited_samples)
        children = {}
        for feature_value in get_feature_values(new_root_1.visited_samples, best_child.feature_index):
            visited = np.array(new_root_1.visited_samples).reshape(-1, 11)
            visited = visited[visited[:, best_child.feature_index] == feature_value]
            new_child = Node(feature_index=root.feature_index, edge_value=feature_value, parent=new_root_1, children={},
                             visited_samples=list(visited))
            new_child.children[list(r.children)[0]] = Node(feature_index=None, edge_value=list(r.children)[0],
                                                           parent=new_child, visited_samples=new_child.visited_samples)
            children[feature_value] = new_child
        new_root_1.children = children
        step2_roots.append(new_root_1)

    # step3
    visited_samples = []
    for r in step2_roots:
        visited_samples += r.visited_samples
    root = Node(feature_index=best_child.feature_index, children={}, visited_samples=visited_samples)
    for r in step2_roots:
        for child in r.children:
            if child not in root.children:
                root.children[child] = r.children[child]
                root.children[child].parent = root
            else:
                root.children[child].children |= r.children[child].children
                root.children[child].visited_samples += r.children[child].visited_samples

    return root


def build_tree():
    current_samples = []
    for idx in range(len(train_data)):
        current_samples.append(train_data[idx, :])
        if calculate_samples_entropy(current_samples) != 0:
            break

    root = insert_root(current_samples)

    current_samples = []
    for idx, data in enumerate(train_data):
        current_samples.append(data)
        parent, node_extend = add_data(root, data)
        if node_extend is not None:
            features_info_gain = calculate_information_gain_branching(node_extend.visited_samples, parent.get_entropy())
            if len(features_info_gain) != 0:  # if there exists a feature for branching
                feature_index = max(features_info_gain,
                                    key=features_info_gain.get)  # index of feature with the highest information gain
                used_features[feature_index] = 0  # you can no longer use this feature!
                # stat filling the new node attributes
                node_extend.feature_index = feature_index
                children = {}
                visited_samples = np.array(node_extend.visited_samples)
                for feature_value in get_feature_values(node_extend.visited_samples, feature_index):
                    visited = visited_samples[visited_samples[:, feature_index] == feature_value]
                    child = Node(edge_value=feature_value, parent=node_extend,
                                 visited_samples=visited.tolist())
                    children[feature_value] = child
                node_extend.children = children

        pull_up(root, current_samples)

    return root


def predict(node, data, find_noise=False):
    global TP, noise, index

    feature_index = node.feature_index
    feature_value = data[feature_index]

    if (node.children[feature_value].feature_index is not None) and (
            node.children[feature_value].omit is not True):  # we are not in a leaf node
        return predict(node.children[feature_value], data, find_noise=find_noise)
    else:  # we are in a leaf node
        predicted = node.get_label()
        if predicted == data[-1]:
            TP = TP + 1
        if find_noise:
            if node.children[feature_value].omit:  # if we have omited its child!
                former_prediction = node.children[feature_value].get_label()
                if former_prediction != predicted and former_prediction == data[-1]:
                    noise.append(index)


def get_nodes(node, level=1):
    global tree_nodes
    if node.children is not None:
        for child in node.children:
            if level in tree_nodes.keys():
                tree_nodes[level].append(node.children[child])
            else:
                tree_nodes[level] = []
                tree_nodes[level].append(node.children[child])
            get_nodes(node.children[child], level + 1)
    return tree_nodes


def get_leaf_nodes():
    leaf = []
    for x in tree_nodes[len(tree_nodes) - 2]:
        if x.feature_index is not None:
            leaf.append(x)
    return leaf


def dataset_preprocess(num_train, num_test):
    train_data = pd.read_csv('dataset/train_data.data', index_col=False)
    test_data = pd.read_csv('dataset/test_data.data', index_col=False)

    train_data.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    test_data.columns = ['S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Label']
    # print(train_data.head())
    train_data = train_data.to_numpy()
    test_data = test_data.to_numpy()

    train_data = train_data[:num_train, :]
    test_data = test_data[:num_test, :]

    return train_data, test_data


if __name__ == "__main__":

    train_data, test_data = dataset_preprocess(500, 50)

    #   storing some useful information
    features = np.arange(train_data.shape[1] - 1)  # all the feature indexes are stored in this array
    used_features = np.ones(train_data.shape[1] - 1)  # this array shows whether a feature has been used or not

    # building a tree from all available features
    root = build_tree()

    # saving the tree structure in a txt file
    rep = repr(root)
    tree_file = open("tree.txt", "w")
    tree_file.write(rep)
    tree_file.close()

    print('built a complete tree')
    TP = 0
    for data in train_data:
        predict(root, data)
    print('accuracy of complete tree', TP / len(train_data))
    print('************')

    # in each iteration, we are going the build a tree from scratch - tree pruning
    num_folds = 4
    noise_data = {}
    accuracies = {}
    features_delete = {}
    for fold in range(num_folds):
        used_features = np.ones(train_data.shape[1] - 1)  # this array shows whether a feature has been used or not
        root = build_tree()

        # fetching all tree nodes
        tree_nodes = {}
        tree_nodes[0] = []
        tree_nodes[0].append(root)
        tree_nodes = get_nodes(root)
        leaf_nodes = get_leaf_nodes()

        #   deleting a fold of leaves
        fold_size = len(leaf_nodes) // num_folds
        delete_nodes = leaf_nodes[fold * fold_size:int(fold * fold_size + fold_size)]

        features_delete[fold] = []
        for node in delete_nodes:
            node.omit = True
            features_delete[fold].append(node.feature_index)

        TP = 0
        noise = []
        for index, data in enumerate(train_data):
            predict(root, data, find_noise=True)

        accuracies[fold] = TP / len(train_data)
        noise_data[fold] = noise
        print('number of noise data:', len(noise_data[fold]))
        print('fold:', fold, 'acc:', TP / len(train_data))
        print('************')

    best_fold = max(accuracies, key=accuracies.get)
    noise = noise_data[best_fold]
    features_delete = features_delete[best_fold]

    train_data, test_data = dataset_preprocess(25000, 1000)

    #uncomment the following lines to delete noise and features
    # print('these features will be deleted:', features_delete)
    # print('these samples will be deleted:', noise)
    # xtrain = np.delete(xtrain, features_delete, 1)
    # xtrain = np.delete(xtrain, noise, 0)
    # xtest = np.delete(xtest, features_delete, 1)

    xtrain = train_data[:, :-1]
    ytrain = train_data[:, -1]
    xtest = test_data[:, :-1]
    ytest = test_data[:, -1]


    model = MLPClassifier(hidden_layer_sizes=(50, 100), activation='relu', solver='adam', max_iter=1500)
    model.fit(xtrain, ytrain)
    train_prediction = model.predict(xtrain)
    test_prediction = model.predict(xtest)

    train_acc = accuracy_score(ytrain, train_prediction) * 100
    test_acc = accuracy_score(ytest, test_prediction) * 100

    print('train_acc: ', train_acc)
    print('test_acc: ', test_acc)
    print("**********")
    print('precision-micro:', precision_score(ytest, test_prediction, average="micro"))
    print('recall-micro:', recall_score(ytest, test_prediction, average="micro"))
    print('f1 score-micro:', f1_score(ytest, test_prediction, average="micro"))
    print("**********")
    print('precision-weighted:', precision_score(ytest, test_prediction, average="weighted", zero_division=0))
    print('recall-weighted:', recall_score(ytest, test_prediction, average="weighted", zero_division=0))
    print('f1 score-weighted:', f1_score(ytest, test_prediction, average="weighted", zero_division=0))
    print("**********")
    print('precision-macro:', precision_score(ytest, test_prediction, average="macro", zero_division=0))
    print('recall-macro:', recall_score(ytest, test_prediction, average="macro", zero_division=0))
    print('f1 score-macro:', f1_score(ytest, test_prediction, average="macro", zero_division=0))
    cm = confusion_matrix(ytest, test_prediction)
    print('confusion matrix:')
    print(cm)

