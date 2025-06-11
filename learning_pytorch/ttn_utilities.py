import torch
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from TorNcon import ncon
from torch.linalg import qr
import numpy as np

class Index():
    
    def __init__(self, dim : int, name = None, id_num = None):
        
        self.dim = dim
        if id_num == None:
            self.id = id(self)
        else:
            self.id = id_num
        if name == None:
            self.name = self.id
        else:
            self.name = name
        
    def __repr__(self):
        
        return f"(name={self.name}, dim={self.dim}, id={self.id % 10000})"
    
    def __eq__(self, other):
        
        return isinstance(other, Index) and self.name == other.name and self.dim == other.dim and self.id == other.id

    def __hash__(self):
        
        return hash((self.name, self.dim))
    
    def copy(self, name = None, id_num = None):
        
        return Index(self.dim, name = name, id_num = id_num)

class Tensor():
    
    def __init__(self, indices, data = None, dtype = torch.float64):
        
        self.indices = indices
        if data == None:
            self.data = self.get_random_tensor(dtype = dtype)
        else:
            self.data = data
        self.shape = self.data.shape
        
        assert [idx.dim for idx in self.indices] == list(self.shape)
        
    def get_random_tensor(self, dtype = torch.float64):
        
        dims = [idx.dim for idx in self.indices]
        return torch.rand(dims, dtype = dtype)
    
    def __repr__(self):
        
        idx_str = ", ".join(repr(idx) for idx in self.indices)
        return f"Tensor(shape={tuple(self.data.shape)}, indices=[{idx_str}])"

class Node():
    
    def __init__(self, tensor : Tensor, parents = None, children = None, name = None, layer = None):
        
        self.tensor = tensor
        self.parents = parents or []
        self.children = children or []
        if name == None:
            self.name = None
        else:
            self.name = str(name)
        self.layer = layer
        
    def __repr__(self):
        
        return (f"Node(name={self.name}, parents={len(self.parents)}, children={len(self.children)}, tensor={repr(self.tensor)}, layer={self.layer})")
        
    def contract(self, node2 : 'Node', parents = None, children = None, name = None, layer = None) -> 'Node':

        node1_indices = self.tensor.indices
        node2_indices = node2.tensor.indices

        # Find shared indices by name
        shared = [idx for idx in node1_indices if idx in node2_indices]
        
        # The output indices are all free indices of both minus shared contracted indices, so just collect free indices from both, excluding shared ones
        free_indices = [idx for idx in node1_indices if idx not in shared] + [idx for idx in node2_indices if idx not in shared]

        # Create mapping for shared indices to positive labels starting from 1
        shared_labels = {idx.id: i+1 for i, idx in enumerate(shared)}  # e.g. {'bond_0': 1}
    
        # Helper to assign labels:
        # - for shared indices: positive label from shared_labels
        # - for unique indices: unique negative labels (start from -1, -2, ...)
        def make_labels(indices, shared_labels, used_negatives):
            
            labels = []
            for idx in indices:
                if idx in shared:
                    labels.append(shared_labels[idx.id])
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
        
        free_labels = []
        for idx in free_indices:
            if idx in node1_indices:
                # pick the label you assigned to that idx in labels1
                free_labels.append(labels1[node1_indices.index(idx)])
            else:
                free_labels.append(labels2[node2_indices.index(idx)])
                
        # Now call ncon with these labels     
        result_data = ncon([self.tensor.data, node2.tensor.data], [labels1, labels2], forder = free_labels)
            
        result_tensor = Tensor(free_indices, result_data)
        result_node = Node(result_tensor, parents = parents, children = children, name = name, layer = layer)
        
        return result_node
        
    def dagger(self):
                
        tensor = self.tensor
        indices = tensor.indices
        data = tensor.data
        
        new_indices = []
        for idx in indices:
            
            if "phys" in idx.name:
                new_indices.append(idx)
            else:
                new_idx = idx.copy(name = idx.name + "'", id_num = idx.id)
                new_indices.append(new_idx)
        
        new_tensor = Tensor(new_indices, torch.conj(data))
                
        new_name = self.name + "'"
                
        node_dagger = Node(new_tensor, self.parents, self.children, new_name, self.layer)
    
        return node_dagger
    
    def prime_bonds(self):
        
        tensor = self.tensor
        indices = tensor.indices
        data = tensor.data
        
        new_indices = []
        for idx in indices:
            
            if "phys" in idx.name:
                new_indices.append(idx)
            else:
                new_idx = idx.copy(name = idx.name + "'", id_num = idx.id)
                new_indices.append(new_idx)
        
        new_tensor = Tensor(new_indices, torch.conj(data))
                                
        node_primed_bonds = Node(new_tensor, self.parents, self.children, self.name, self.layer)
    
        return node_primed_bonds
        
class TTN():
    
    def __init__(self, nodes = None):
        
        self.nodes = nodes or []
        
    def __repr__(self):
        
        return (f"TTN(total_nodes={len(self.nodes)}):\n" +
                "\n".join(f"  {repr(node)}" for node in self.nodes))
        
    def visualize(self):
    
        root_node = [node for node in self.nodes if not node.parents][0]
        
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
                edge_labels = ",".join([f'{idx.name},dim={idx.dim}' for idx in shared_indices])

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
        
    def get_physical_indices(self):
        
        """
        Returns a list of all physical indices used in the TTN.
        Physical indices are identified by their name starting with 'phys_'.
        These indices usually appear at the leaves of the TTN.
        """
        
        phys_indices = []
        # Collect all unique physical indices from all nodes
        seen = set()
        for node in self.nodes:
            for idx in node.tensor.indices:
                if idx.name.startswith("phys_") and idx not in seen:
                    phys_indices.append(idx)
                    seen.add(idx)
        return phys_indices
   
    def canonicalize(self, normalize = False):

        # Group nodes by layer
        layer_dict = {}
        for node in self.nodes:
            layer_dict.setdefault(node.layer, []).append(node)
            
        max_layer = max(layer_dict.keys())
        
        for layer in range(max_layer + 1):
                                                            
            for node in layer_dict[layer]:
                                                                                                                
                tensor = node.tensor
                data = tensor.data
                
                if node.parents:
                    
                    parent = node.parents[0]
                    
                    top_idx = next(idx for idx in tensor.indices if idx in node.parents[0].tensor.indices)
                    bottom_idx_list = [idx for idx in tensor.indices if idx != top_idx]
                    
                    # QR decomposition: reshape -> QR -> reshape
                    matrix = data.reshape(-1, top_idx.dim)
                    Q, R = qr(matrix)
                        
                    # New bond dimension after QR
                    k = Q.shape[1]
                    new_bond = Index(k, name = top_idx.name)
                    Q_tensor = Q.reshape(*[element.dim for element in bottom_idx_list], k)

                    # Replace current node's tensor with Q
                    node.tensor = Tensor(bottom_idx_list + [new_bond], Q_tensor)

                    # Build R tensor and contract it into parent (always, even for root)
                    R_tensor = Tensor([new_bond, top_idx], R)
                    R_node = Node(R_tensor)
                                    
                    # new_parent = parent.contract(R_node) # TODO: understand why this instead of the line below changes the norm
                    new_parent = R_node.contract(parent)
                    parent.tensor = new_parent.tensor # Replace parent’s tensor with contracted result
                    
                elif normalize: # case of the root node
                                        
                    bottom_idx_list = tensor.indices

                    # QR decomposition: reshape -> QR -> reshape
                    matrix = data.reshape(-1, 1)
                    Q, R = qr(matrix)

                    # New bond dimension after QR
                    Q_tensor = Q.reshape(*[element.dim for element in bottom_idx_list])

                    # Replace current node's tensor with Q
                    node.tensor = Tensor(bottom_idx_list, Q_tensor)
                    
        return self
                  
    def norm(self):
        
        # Group nodes by layer
        layer_dict = {}
        for node in self.nodes:
            layer_dict.setdefault(node.layer, []).append(node)
            
        prev_res_layer = []
        layer = 0 
                
        for node in layer_dict[layer]:
            
            node_dag = node.dagger()
            
            prev_res_layer.append(node.contract(node_dag))
                        
        for layer in range(1, max(layer_dict.keys()) + 1):
            
            res_layer = []
            
            for i, node in enumerate(layer_dict[layer]):    
                
                left, right = prev_res_layer[2*i], prev_res_layer[2*i+1]
                top, bottom = node, node.dagger()
                
                tmp = top.contract(left)                
                tmp = tmp.contract(right)
                tmp = tmp.contract(bottom)
                res_layer.append(tmp)
                
            prev_res_layer = res_layer      
            
        return torch.sqrt(prev_res_layer[0].tensor.data)
    
    def contract(self, ttn2):
        
        """
        Contract two TTNs. Assumes both TTNs have the same structure.
        """

        # Step 1: Build layer dictionary for both TTNs
        layer_dict_1 = {}
        layer_dict_2 = {}

        for node in self.nodes:
            layer_dict_1.setdefault(node.layer, []).append(node)
        for node in ttn2.nodes:
            layer_dict_2.setdefault(node.layer, []).append(node.dagger())

        # Step 2: Leaf layer (assumes same number of leaves in same order)
        prev_res_layer = []
        layer = 0

        for node1, node2 in zip(layer_dict_1[layer], layer_dict_2[layer]):
            result = node1.contract(node2)
            prev_res_layer.append(result)

        # Step 3: Internal layers
        for layer in range(1, max(layer_dict_1.keys()) + 1):
            res_layer = []

            for i, (node1, node2) in enumerate(zip(layer_dict_1[layer], layer_dict_2[layer])):
                left = prev_res_layer[2 * i]
                right = prev_res_layer[2 * i + 1]

                tmp = node1.contract(left)
                tmp = tmp.contract(right)
                tmp = tmp.contract(node2)
                res_layer.append(tmp)

            prev_res_layer = res_layer

        return prev_res_layer[0]  # the final node from the top contraction

def random_binary_ttn_old(depth, D, d) -> TTN:
    
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
        root_node = Node(tensor, parents=[], children=[], name = len(nodes), layer = 0)
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
            bond_idx = Index(np.random.randint(1, 10), name=f"bond_{len(nodes)}") # TODO: replace np.random.randint(1, 10) with D after testing is finished
            tensor = Tensor([phys_idx_left, phys_idx_right, bond_idx])
            leaf_node = Node(tensor, name = len(nodes), layer = layer)
            nodes.append(leaf_node)
            
            return leaf_node, bond_idx

        left_node, left_bond_idx = build_subtree(layer - 1)
        right_node, right_bond_idx = build_subtree(layer - 1)

        if layer == depth - 1: # case of root node (stopping condition of recursion)
            
            tensor = Tensor([left_bond_idx, right_bond_idx])
            root_node = Node(tensor, parents=[], children=[left_node, right_node], name = len(nodes), layer = layer)
            left_node.parents.append(root_node)
            right_node.parents.append(root_node)
            nodes.append(root_node)
            return root_node, None
        
        else:
            
            parent_bond_idx = Index(np.random.randint(1, 10), name=f"bond_{len(nodes)}") # TODO: replace np.random.randint(1, 10) with D after testing is finished
            tensor = Tensor([left_bond_idx, right_bond_idx, parent_bond_idx])
            int_node = Node(tensor, parents=[], children=[left_node, right_node], name = len(nodes), layer = layer)
            left_node.parents.append(int_node)
            right_node.parents.append(int_node)
            nodes.append(int_node)
            
            return int_node, parent_bond_idx

    # Build full TTN
    _, _ = build_subtree(depth - 1)
    
    return TTN(nodes)

def random_binary_ttn(N, d, D, physical_indices=None):
    """
    Creates a binary TTN with `N` physical indices (must be power of 2).

    - d: physical dimension
    - D: bond dimension (used for internal bonds)
    - physical_indices: optional list of Index objects to use at the leaves

    Returns a TTN where each node has proper parents, children, name, and layer set.
    """
    assert (N & (N - 1)) == 0 and N > 0, "N must be a power of 2 and > 0"

    # Create or reuse physical indices
    if physical_indices is not None:
        assert len(physical_indices) == N, "Provided physical_indices length must match N"
        phys = physical_indices
    else:
        phys = [Index(d, name=f"phys_{i}") for i in range(N)]

    nodes = []
    current_nodes = []
    current_layer = 0
    node_id = 0

    # Leaf layer
    for i in range(0, N, 2):
        bond = Index(D, name=f"bond_{node_id}")
        tensor = Tensor([phys[i], phys[i + 1], bond])
        node = Node(tensor, parents=[], children=[], name=node_id, layer=current_layer)
        current_nodes.append(node)
        nodes.append(node)
        node_id += 1

    # Internal layers
    while len(current_nodes) > 1:
        next_nodes = []
        current_layer += 1

        for i in range(0, len(current_nodes), 2):
            left = current_nodes[i]
            right = current_nodes[i + 1]

            if len(current_nodes) == 2:
                # This is the final layer — root node: only two bond indices
                tensor = Tensor([left.tensor.indices[-1], right.tensor.indices[-1]])
            else:
                bond = Index(D, name=f"bond_{node_id}")
                tensor = Tensor([left.tensor.indices[-1], right.tensor.indices[-1], bond])

            parent = Node(tensor, parents=[], children=[left, right], name=node_id, layer=current_layer)
            left.parents.append(parent)
            right.parents.append(parent)

            next_nodes.append(parent)
            nodes.append(parent)
            node_id += 1

        current_nodes = next_nodes

    return TTN(nodes=nodes)
