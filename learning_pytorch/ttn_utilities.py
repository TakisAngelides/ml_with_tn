import torch
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from TorNcon import ncon

class Index():
    
    def __init__(self, dim : int, name = None):
        
        self.dim = dim
        self.name = name or id(self)
        
    def __repr__(self):
        
        return f"(name={self.name}, dim={self.dim})"
    
    def __eq__(self, other):
        
        return isinstance(other, Index) and self.name == other.name and self.dim == other.dim

    def __hash__(self):
        
        return hash((self.name, self.dim))

class Tensor():
    
    def __init__(self, indices, data = None):
        
        self.indices = indices
        if data == None:
            self.data = self.get_random_tensor()
        else:
            self.data = data
        
    def get_random_tensor(self):
        
        dims = [idx.dim for idx in self.indices]
        return torch.rand(dims)
    
    def __repr__(self):
        
        idx_str = ", ".join(repr(idx) for idx in self.indices)
        return f"Tensor(shape={tuple(self.data.shape)}, indices=[{idx_str}])"

class Node():
    
    def __init__(self, tensor : Tensor, parents = None, children = None, name = None):
        
        self.tensor = tensor
        self.parents = parents or []
        self.children = children or []
        self.name = name
        
    def __repr__(self):
        
        return (f"Node(name={self.name}, "
                f"parents={len(self.parents)}, "
                f"children={len(self.children)}, "
                f"tensor={repr(self.tensor)})")
        
class TTN():
    
    def __init__(self, nodes = None):
        
        self.nodes = nodes or []
        
    def __repr__(self):
        
        return (f"TTN(total_nodes={len(self.nodes)}):\n" +
                "\n".join(f"  {repr(node)}" for node in self.nodes))
        
def random_binary_ttn(depth, D, d) -> TTN:
    
    """
    Builds a perfect binary-tree tensor network of given `depth`.
    - depth: number of layers (leaves are at layer 0; root is at layer depth-1)
    - D: bond dimension for all internal bonds
    - d: physical dimension (only leaves have two physical indices of dimension d)
    
    Notes:
    
    Leaves are at the bottom (depth = 0),
    Internal nodes are in the middle (depth = 1 to depth-2),
    Root is at the top (depth = depth - 1).
    
    Returns a TTN object whose `nodes` list contains all Node instances.
    """
    
    nodes = []
    site_counter = [0]

    # Edge case: depth must be at least 1
    if depth < 1:
        raise ValueError("Depth must be >= 1.")

    # Special case: depth == 1 → single leaf node (also the root)
    if depth == 1:
        
        phys_idx_left  = Index(d, name=f"phys_{2*site_counter[0]}")
        phys_idx_right = Index(d, name=f"phys_{2*site_counter[0]+1}")
        tensor = Tensor([phys_idx_left, phys_idx_right])
        root_node = Node(tensor, parents=[], children=[], name = len(nodes))
        nodes.append(root_node)
        return TTN(nodes)

    def build_subtree(layer):
                        
        """
        Recursively build a subtree rooted at the specified layer.
        
        Note: This internal helper recursively constructs the binary tree starting from the leaves (layer = 0) up to the root.
        
        """
        
        if layer == 0: # case of leafs
            
            phys_idx_left  = Index(d, name=f"phys_{2*site_counter[0]}")
            phys_idx_right = Index(d, name=f"phys_{2*site_counter[0]+1}")
            site_counter[0] += 1
            bond_idx = Index(D, name=f"bond_{len(nodes)}")
            tensor = Tensor([phys_idx_left, phys_idx_right, bond_idx])
            leaf_node = Node(tensor, name = len(nodes))
            nodes.append(leaf_node)
            
            return leaf_node, bond_idx

        left_node, left_bond_idx = build_subtree(layer - 1)
        right_node, right_bond_idx = build_subtree(layer - 1)

        if layer == depth - 1: # case of root node (stopping condition of recursion)
            
            tensor = Tensor([left_bond_idx, right_bond_idx])
            root_node = Node(tensor, parents=[], children=[left_node, right_node], name = len(nodes))
            left_node.parents.append(root_node)
            right_node.parents.append(root_node)
            nodes.append(root_node)
            return root_node, None
        
        else:
            
            parent_bond_idx = Index(D, name=f"bond_{len(nodes)}")
            tensor = Tensor([left_bond_idx, right_bond_idx, parent_bond_idx])
            int_node = Node(tensor, parents=[], children=[left_node, right_node], name = len(nodes))
            left_node.parents.append(int_node)
            right_node.parents.append(int_node)
            nodes.append(int_node)
            
            return int_node, parent_bond_idx

    # Build full TTN
    root, _ = build_subtree(depth - 1)
    
    return TTN(nodes)

def visualize_ttn(ttn : TTN):
    
    root_node = [node for node in ttn.nodes if not node.parents][0]
    
    G = nx.DiGraph()  # Directional to reflect parent-child structure
    visited = set()

    def node_label(node):
        return node.name

    def add_edges_recursive(node, depth=0):
        node_id = node_label(node)
        if node_id in visited:
            return
        visited.add(node_id)

        G.add_node(node_id, depth=depth)

        for child in node.children:
            child_id = node_label(child)
            # Find shared indices between parent and child tensors
            shared_indices = [idx for idx in node.tensor.indices if idx in child.tensor.indices]
            edge_labels = ", ".join(idx.name or str(idx) for idx in shared_indices)

            G.add_edge(node_id, child_id, label=edge_labels)
            add_edges_recursive(child, depth + 1)

    add_edges_recursive(root_node)

    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Grankdir=TB')
    except Exception:
        print("Warning: pygraphviz/Graphviz not found. Using spring_layout instead.")
        pos = nx.spring_layout(G)

    # Node colors based on depth
    depths = nx.get_node_attributes(G, 'depth')
    max_depth = max(depths.values()) + 1 if depths else 1
    cmap = plt.get_cmap('Greens')
    node_colors = [cmap(depth / max_depth) for depth in depths.values()]

    plt.figure(figsize=(12, 8))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        nx.draw(
            G,
            pos=pos,
            labels={n: n for n in G.nodes()},
            with_labels=True,
            node_size=3000,
            node_color=node_colors,
            font_size=10,
            font_weight='bold',
            edge_color='gray'
        )
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')

    plt.title("Tree Tensor Network")
    plt.show()

def contract(node1 : Node, node2 : Node, parents = None, children = None, name = None) -> Node:

    node1_indices = node1.tensor.indices
    node2_indices = node2.tensor.indices

    # Find shared indices by name
    shared = [idx for idx in node1_indices if idx.name in {ind.name for ind in node2_indices}]

    # Create mapping for shared indices to positive labels starting from 1
    shared_labels = {idx.name: i+1 for i, idx in enumerate(shared)}  # e.g. {'bond_0': 1}

    # Helper to assign labels:
    # - for shared indices: positive label from shared_labels
    # - for unique indices: unique negative labels (start from -1, -2, ...)
    def make_labels(indices, shared_labels, used_negatives):
        
        labels = []
        for idx in indices:
            if idx.name in shared_labels:
                labels.append(shared_labels[idx.name])
            else:
                # assign new negative label not used before
                if used_negatives:
                    neg_label = min(used_negatives) - 1
                else:
                    neg_label = -1
                used_negatives.add(neg_label)
                labels.append(neg_label)

        return labels

    used_negatives = set()
    labels1 = make_labels(node1_indices, shared_labels, used_negatives) # modifies used_negatives
    labels2 = make_labels(node2_indices, shared_labels, used_negatives)

    # Now call ncon with these labels
    result_data = ncon([node1.tensor.data, node2.tensor.data], [labels1, labels2])

    # The output indices are all free indices of both minus shared contracted indices
    # So just collect free indices from both, excluding shared ones
    free_indices = [idx for idx in node1_indices if idx.name not in shared_labels] + \
                   [idx for idx in node2_indices if idx.name not in shared_labels]
                   
    result_tensor = Tensor(free_indices, result_data)
    result_node = Node(result_tensor, parents = parents, children = children, name = name)
    
    return result_node




ttn = random_binary_ttn(depth=2, D=1, d=2)





