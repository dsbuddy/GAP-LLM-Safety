import networkx as nx
import matplotlib.pyplot as plt
import random

# Function to get a random color based on the tree number and level
def random_color(tree_num, level):
    available_colors = ['lightgreen', 'lightsalmon', 'lightblue', 'plum'][:tree_num]
    if level == 0:
        return available_colors[tree_num-1]  # For the root node, return the first color
    else:
        return random.choice(available_colors)

# Function to create a full binary tree
def create_full_binary_tree(depth, tree_num=0, is_color=False, aggregation_prob=0.2, pruning_prob=0.1):
    tree = nx.DiGraph()
    if is_color:
        tree.add_node(0, color=random_color(tree_num, 0))  # Root node with a color based on the tree number
        last_level = [(0, tree.nodes[0]['color'])]  # List to keep track of nodes and their colors in the current level
    else:
        tree.add_node(0)  # Root node
        last_level = [0]  # List to keep track of nodes in the current level

    for level in range(1, depth):
        new_level = []
        if is_color:
            for node, color in last_level:
                # Decide whether to prune the branch
                if random.random() < pruning_prob:
                    continue  # Skip this node and move to the next one

                # Decide whether to create two new nodes or aggregate with an existing node
                if random.random() < aggregation_prob and new_level:
                    # Aggregate with an existing node
                    existing_node = random.choice(new_level)
                    tree.add_edge(node, existing_node[0])
                else:
                    # Create two new nodes
                    left_child = 2 * node + 1
                    right_child = 2 * node + 2
                    tree.add_node(left_child, color=random_color(tree_num, level + 1))
                    tree.add_node(right_child, color=random_color(tree_num, level + 1))
                    tree.add_edge(node, left_child)
                    tree.add_edge(node, right_child)
                    new_level.append((left_child, tree.nodes[left_child]['color']))
                    new_level.append((right_child, tree.nodes[right_child]['color']))
        else:
            for node in last_level:
                # Decide whether to prune the branch (only after level 2)
                if level >= 3 and random.random() < pruning_prob:
                    continue  # Skip this node and move to the next one

                # Decide whether to create two new nodes or aggregate with an existing node
                if random.random() < aggregation_prob and new_level:
                    # Aggregate with an existing node
                    existing_node = random.choice(new_level)
                    tree.add_edge(node, existing_node)
                else:
                    # Create two new nodes
                    left_child = 2 * node + 1
                    right_child = 2 * node + 2
                    tree.add_node(left_child)
                    tree.add_node(right_child)
                    tree.add_edge(node, left_child)
                    tree.add_edge(node, right_child)
                    new_level.append(left_child)
                    new_level.append(right_child)
        last_level = new_level

    return tree

# Function to visualize the trees
def visualize_trees_tap(trees, labels, is_color=False, node_size=100):
    fig, axes = plt.subplots(1, 4, figsize=(24, 4))
    colors = ['lightgreen', 'lightsalmon', 'lightblue', 'plum']  # List of colors for each tree

    for i, (tree, label) in enumerate(zip(trees, labels)):
        pos = nx.nx_pydot.graphviz_layout(tree, prog='dot', root=0)
        if is_color:
            nx.draw(tree, pos, with_labels=False, node_color=colors[i], node_size=node_size, ax=axes[i])
        else:
            nx.draw(tree, pos, with_labels=False, node_color='lightgray', node_size=node_size, ax=axes[i])
        axes[i].set_title(label, fontsize=28, y=1.05)
        axes[i].axis('off')

    plt.tight_layout()
    if is_color:
        plt.savefig('trees_tap_color_small_g.pdf', dpi=300, bbox_inches='tight')
    else:
        plt.savefig('trees_tap_orig_small_g.pdf', dpi=300, bbox_inches='tight')


# Function to visualize the trees
def visualize_trees_gap(trees, labels, is_color=False, node_size=100, count=1):
    fig, axes = plt.subplots(1, 4, figsize=(24, 4))
    max_height = 6  # Set the maximum height for the trees
    for i, (tree, label) in enumerate(zip(trees, labels)):
        pos = nx.nx_pydot.graphviz_layout(tree, prog='dot', root=0)
        if is_color:
            node_colors = [tree.nodes[node]['color'] for node in tree.nodes()]
            nx.draw(tree, pos, with_labels=False, node_color=node_colors, node_size=node_size, ax=axes[i])
        else:
            nx.draw(tree, pos, with_labels=False, node_color='lightgray', node_size=node_size, ax=axes[i])
        axes[i].set_title(label, fontsize=28, y=1.05)
        axes[i].axis('off')
        axes[i].set_ylim(max_height-i*50)  # Set the y-axis limits

    plt.tight_layout()
    if is_color:
        plt.savefig('trees_gap_color_small_g_{}.pdf'.format(count), dpi=300, bbox_inches='tight')
    else:
        plt.savefig('trees_gap_orig_small_g.pdf', dpi=300, bbox_inches='tight')

node_size = 150
num_trees = 4

# Create TAP Trees
max_depth = 6
# trees_tap = [create_full_binary_tree(max_depth) for _ in range(num_trees)]

# Create GAP Trees
depths_gap = [max_depth-i for i in range(num_trees)]
# trees_gap = [create_full_binary_tree(depth) for depth in depths_gap]
# trees_gap_color = [create_full_binary_tree(depth, tree_num=i+1, is_color=True) for i, depth in enumerate(depths_gap)]


trees_tap = [create_full_binary_tree(max_depth, aggregation_prob=0.0, pruning_prob=0.1) for _ in range(num_trees)]
for m in range (1,100):
    trees_gap = [create_full_binary_tree(depth, tree_num=i+1, is_color=True, aggregation_prob=0.20, pruning_prob=0.1) for i, depth in enumerate(depths_gap)]
    labels = [f'Seed #{i+1}' for i in range(num_trees)]
    visualize_trees_gap(trees_gap, labels, is_color=True, node_size=node_size, count=m)
# trees_gap = [create_full_binary_tree(depth, tree_num=i+1, is_color=True, aggregation_prob=0.15, pruning_prob=0.1) for i, depth in enumerate(depths_gap)]

# Visualize the trees with different labels and node colors
labels = [f'Seed #{i+1}' for i in range(num_trees)]
visualize_trees_tap(trees_tap, labels, node_size=node_size)
visualize_trees_tap(trees_tap, labels, is_color=True, node_size=node_size)
visualize_trees_gap(trees_gap, labels, node_size=node_size)
visualize_trees_gap(trees_gap, labels, is_color=True, node_size=node_size)

