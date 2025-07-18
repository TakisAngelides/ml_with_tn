import torch
import networkx as nx
import matplotlib.pyplot as plt
import warnings
from TorNcon import ncon
from torch.linalg import qr
import numpy as np
from typing import List, Optional, Callable
from collections import deque
torch.manual_seed(42)

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
    
    def prime(self):
        
        return Index(self.dim, name = self.name + "'", id_num = self.id)

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
    
    def unprime_physical(self):
        
        tensor = self.tensor
        indices = tensor.indices
        data = tensor.data
        
        new_indices = []
        for idx in indices:
            
            if "phys" in idx.name and idx.name[-1] == "'":
                new_idx = idx.copy(name = idx.name[:-1], id_num = idx.id)
                new_indices.append(new_idx)
            else:
                new_indices.append(idx)
        
        new_tensor = Tensor(new_indices, data)
                                
        res = Node(new_tensor, self.parents, self.children, self.name, self.layer)
                
        return res
    
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
   
    def canonicalize(self, target_layer = None, target_node_idx = 0, normalize=False):

        # Group nodes by layer
        layer_dict = {}
        for node in self.nodes:
            layer_dict.setdefault(node.layer, []).append(node)
                
        max_layer = max(layer_dict.keys())
        if target_layer == None:
            target_layer = max_layer     

        # Select target node
        target_node = layer_dict[target_layer % (max_layer+1)][target_node_idx % len(layer_dict[target_layer % (max_layer+1)])]
        
        # Compute distances to target node using BFS
        distances = {}
        queue = deque([(target_node, 0)])
        visited = set()
        
        while queue:
                        
            current_node, dist = queue.popleft()
            if current_node in visited:
                continue
            visited.add(current_node)
            distances[current_node] = dist

            # Add neighbors
            for parent in getattr(current_node, "parents", []):
                if parent not in visited:
                    queue.append((parent, dist+1))
            for child in getattr(current_node, "children", []):
                if child not in visited:
                    queue.append((child, dist+1))
                    
        # Sort nodes by distance (descending)
        sorted_nodes = sorted(distances.items(), key=lambda x: -x[1])
        
        for node, dist in sorted_nodes:
            tensor = node.tensor
            data = tensor.data

            if node == target_node: # Special case for target node
                if normalize:
                    bottom_idx_list = tensor.indices
                    matrix = data.reshape(-1, 1)
                    Q, R = qr(matrix)
                    Q_tensor = Q.reshape(*[element.dim for element in bottom_idx_list])
                    node.tensor = Tensor(bottom_idx_list, Q_tensor)
                continue

            # Determine contraction direction (move R towards parent or child)
            if node.parents and distances.get(node.parents[0], float('inf')) < dist: # do I have a parent and is my parent closer to the target?
                # Contract into parent
                parent = node.parents[0]
                top_idx = next(idx for idx in tensor.indices if idx in parent.tensor.indices)
                bottom_idx_list = [idx for idx in tensor.indices if idx != top_idx]

                matrix = data.reshape(-1, top_idx.dim)
                Q, R = qr(matrix)

                k = Q.shape[1]
                new_bond = Index(k, name=top_idx.name)
                Q_tensor = Q.reshape(*[element.dim for element in bottom_idx_list], k)
                node.tensor = Tensor(bottom_idx_list + [new_bond], Q_tensor)

                R_tensor = Tensor([new_bond, top_idx], R)
                R_node = Node(R_tensor)
                new_parent = R_node.contract(parent)
                parent.tensor = new_parent.tensor

            elif getattr(node, "children", []) and distances.get(node.children[0], float('inf')) < dist:
                # Contract into child
                child = node.children[0]
                top_idx = next(idx for idx in tensor.indices if idx in child.tensor.indices)
                bottom_idx_list = [idx for idx in tensor.indices if idx != top_idx]

                matrix = data.reshape(-1, top_idx.dim)
                Q, R = qr(matrix)

                k = Q.shape[1]
                new_bond = Index(k, name=top_idx.name)
                Q_tensor = Q.reshape(*[element.dim for element in bottom_idx_list], k)
                node.tensor = Tensor(bottom_idx_list + [new_bond], Q_tensor)

                R_tensor = Tensor([new_bond, top_idx], R)
                R_node = Node(R_tensor)
                new_child = R_node.contract(child)
                child.tensor = new_child.tensor

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
            
        return torch.sqrt(prev_res_layer[0].tensor.data).item()
    
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

        return prev_res_layer[0].tensor.data.item()  # the final node from the top contraction

    def expectation_value(self, mpo):
        
        # Group nodes by layer
        layer_dict = {}
        for node in self.nodes:
            layer_dict.setdefault(node.layer, []).append(node)
            
        prev_res_layer = []
        layer = 0 
                    
        for i, node in enumerate(layer_dict[layer]):
                        
            node_dag = node.dagger()
                        
            prev_res_layer.append(node.contract(mpo[2*i].contract(mpo[2*i+1])).unprime_physical().contract(node_dag))
                        
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
                        
        return prev_res_layer[0].tensor.data.item()/self.contract(self)
    
class MPO():
    
    def __init__(self, n: int, d: int, builder_fn: Optional[Callable[[int], List[Tensor]]] = None, nodes: Optional[List[Node]] = None):
        """
        Matrix Product Operator (MPO) class using Nodes and Tensors.
        """
        self.n = n
        self.d = d
        
        if nodes is not None:
            self.nodes = nodes
        elif builder_fn is not None:
            tensors = builder_fn(n)
            self.nodes = [Node(tensor) for tensor in tensors]
        else:
            raise ValueError("Either builder_fn or nodes must be provided.")

    def __repr__(self):
        nodes_str = "\n".join([repr(node.tensor) for node in self.nodes])
        return f"MPO(n={self.n}, nodes=\n{nodes_str})"

    def __getitem__(self, idx):
        return self.nodes[idx]

    def __setitem__(self, idx, value):
        self.nodes[idx] = value

    def get_linkdims(self):
        linkdims = []
        for node in self.nodes:
            bond_indices = [idx for idx in node.tensor.indices if not idx.name.startswith("phys_")]
            linkdims.append([idx.dim for idx in bond_indices])
        dims = [1] + [dims[1] for dims in linkdims] + [1]
        return dims

    def mpo_to_matrix(self):
        """
        Converts the MPO to a full dense matrix.
        """
        n = self.n
        d = self.d

        tmp = torch.tensor([1.0], dtype=torch.float64)
        res = torch.tensordot(self.nodes[-1].tensor.data, tmp, dims=([1], [0]))
        for i in reversed(range(n - 1)):
            res = torch.tensordot(self.nodes[i].tensor.data, res, dims=([1], [0]))
        res = torch.tensordot(tmp, res, dims=([0], [0]))

        perm = torch.cat([torch.arange(1, 2 * n, 2), torch.arange(0, 2 * n, 2)]).tolist()
        res = res.permute(*perm).reshape(d ** n, d ** n)
        return res

    def compress(self, tol=1e-14, maxD=np.inf):
        """
        Simple MPO compression with QR + SVD, using Nodes.
        """
        n = self.n
        res = []

        # Left canonicalization via QR
        for i in range(n - 1):
            tensor_i = self.nodes[i].tensor
            Dl, Dr, d1, d2 = tensor_i.data.shape
            M = tensor_i.data.permute(0, 2, 3, 1).reshape(Dl * d1 * d2, Dr)
            Q, R = torch.linalg.qr(M)
            newD = Q.shape[1]
            Q_tensor = Q.reshape(Dl, d1, d2, newD).permute(0, 3, 1, 2)

            # Update indices
            left_bond = tensor_i.indices[0]
            phys_in, phys_out = tensor_i.indices[1], tensor_i.indices[2]
            new_right_bond = Index(newD, name=tensor_i.indices[3].name)

            res.append(Node(Tensor([left_bond, new_right_bond, phys_in, phys_out], Q_tensor)))

            tensor_next = self.nodes[i + 1].tensor
            next_R = torch.tensordot(R, tensor_next.data.reshape(Dr, -1), [[1], [0]]).reshape(newD, *tensor_next.data.shape[1:])
            new_indices = [new_right_bond] + tensor_next.indices[1:]
            self.nodes[i + 1] = Node(Tensor(new_indices, next_R))

        res.append(self.nodes[-1])

        # Right to left SVD truncation
        for i in reversed(range(1, n)):
            tensor_i = res[i].tensor
            Dl, Dr, d1, d2 = tensor_i.data.shape
            M = tensor_i.data.reshape(Dl, Dr * d1 * d2)
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            keep = min(int((S.abs() > tol).sum().item()), maxD)
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
            V_tensor = Vh.reshape(keep, Dr, d1, d2)
            new_right_bond = Index(keep, name=tensor_i.indices[1].name)
            res[i] = Node(Tensor([new_right_bond, tensor_i.indices[1], tensor_i.indices[2], tensor_i.indices[3]], V_tensor))

            B = U @ torch.diag(S)
            prev_tensor = res[i - 1].tensor
            C = torch.tensordot(prev_tensor.data, B, [[1], [0]])
            new_indices = [prev_tensor.indices[0], Index(keep, name=prev_tensor.indices[1].name),
                           prev_tensor.indices[2], prev_tensor.indices[3]]
            res[i - 1] = Node(Tensor(new_indices, C.permute(0, 3, 1, 2)))

        return MPO(n=self.n, d=self.d, nodes=res)

    def add_mpo(self, mpo2):
        """
        Sum two MPOs. Assumes compatible structures.
        """
        assert self.n == mpo2.n
        assert self.d == mpo2.d
        d = self.d
        n = self.n

        res_nodes = []
        for i in range(n):
            t1 = self.nodes[i].tensor
            t2 = mpo2.nodes[i].tensor

            Dl1, Dr1 = t1.indices[0].dim, t1.indices[1].dim
            Dl2, Dr2 = t2.indices[0].dim, t2.indices[1].dim

            new_Dl, new_Dr = Dl1 + Dl2, Dr1 + Dr2

            new_left_idx = Index(new_Dl, name=t1.indices[0].name)
            new_right_idx = Index(new_Dr, name=t1.indices[1].name)
            phys_in, phys_out = t1.indices[2], t1.indices[3]

            new_tensor_data = torch.zeros((new_Dl, new_Dr, d, d), dtype=t1.data.dtype)

            new_tensor_data[:Dl1, :Dr1, :, :] = t1.data
            new_tensor_data[Dl1:, Dr1:, :, :] = t2.data

            new_tensor = Tensor([new_left_idx, new_right_idx, phys_in, phys_out], new_tensor_data)
            res_nodes.append(Node(new_tensor))

        return MPO(n=n, d=d, nodes=res_nodes)

class DMRG:
    
    def __init__(self, mpo : MPO, ttn : TTN, sweeps = 10, etol = 1e-14, maxD = np.inf, tol = 1e-14, compress = True, verbose = False):
        
        self.mpo = mpo
        self.ttn = ttn 
        self.sweeps = sweeps
        self.etol = etol
        self.maxD = maxD
        self.tol = tol
        self.compress = compress
        self.verbose = verbose
        self.N = self.mps.N
        self.env = self._initialize_env()

    def _initialize_env(self):
        
        env = [torch.tensor([[[1.0]]], dtype=torch.float64)] + [None for _ in range(self.N)] + [torch.tensor([[[1.0]]], dtype=torch.float64)]
        for i in reversed(range(self.N)):
            tmp = torch.tensordot(env[i + 2], self.mps[i], [[0], [1]])
            tmp = torch.tensordot(tmp, self.mpo[i], [[0, 3], [1, 2]])
            env[i + 1] = torch.tensordot(tmp, self.mps[i].conj(), [[0, 3], [1, 2]])
        return env

    def run(self):
        
        E_curr = 1e-16

        for sweep in range(self.sweeps):
            
            E = self._sweep_lr(sweep)
            E = self._sweep_rl(sweep)
            frac_change = abs((E - E_curr) / E_curr)
            E_curr = E

            print(f"Sweep {sweep}, E = {E}")
            if frac_change < self.etol:
                print("Energy accuracy reached.")
                return E_curr, self.mps

        print("Maximum sweeps reached before reaching desired energy accuracy.")
        return E_curr, self.mps

    def _sweep_lr(self, sweep):
        for i in range(self.N - 1):
            Heff, dims = self._build_Heff(i)
            evals, evecs = torch.linalg.eigh(Heff)
            E = evals[0].item()
            gs = evecs[:, 0]
            self._update_mps(i, gs, dims)
            self._update_env_left(i)
            if self.verbose:
                print(f"Sweep {sweep} L→R, site {i}, E = {E}")
        return E

    def _sweep_rl(self, sweep):
        for i in reversed(range(self.N - 1)):
            Heff, dims = self._build_Heff(i)
            evals, evecs = torch.linalg.eigh(Heff)
            E = evals[0].item()
            gs = evecs[:, 0]
            self._update_mps(i, gs, dims, right_to_left=True)
            self._update_env_right(i)
            if self.verbose:
                print(f"Sweep {sweep} R→L, site {i}, E = {E}")
        return E

    def _build_Heff(self, i):
        L = self.env[i]
        R = self.env[i + 3]
        W = torch.tensordot(self.mpo[i], self.mpo[i + 1], [[1], [0]])
        LMW = torch.tensordot(L, W, [[1], [0]])
        full = torch.tensordot(LMW, R, [[4], [1]])
        full = full.permute(0, 2, 4, 6, 1, 3, 5, 7)
        dims = full.shape
        Heff = full.reshape(
            dims[0] * dims[1] * dims[2] * dims[3],
            dims[4] * dims[5] * dims[6] * dims[7]
        )
        return Heff, dims

    def _update_mps(self, i, gs, dims, right_to_left=False):
        gs = gs.reshape(dims[0] * dims[1], dims[2] * dims[3])
        u, s, vh = torch.linalg.svd(gs, full_matrices=False)

        if self.compress:
            D = min((torch.abs(s) > self.tol).sum().item(), self.maxD)
        else:
            D = (torch.abs(s) > 1e-14).sum().item()

        u, s, vh = u[:, :D], s[:D], vh[:D, :]

        if not right_to_left:
            left = u.reshape(dims[0], dims[1], D).permute(0, 2, 1)
            vh = vh.reshape(D, dims[2], dims[3]).permute(0, 2, 1)
            right = torch.tensordot(torch.diag(s), vh, [[1], [0]])
            self.mps[i], self.mps[i + 1] = left, right
        else:
            u = u.reshape(dims[0], dims[1], D).permute(0, 2, 1)
            left = torch.tensordot(u, torch.diag(s), [[1], [0]]).permute(0, 2, 1)
            right = vh.reshape(D, dims[2], dims[3]).permute(0, 2, 1)
            self.mps[i], self.mps[i + 1] = left, right

    def _update_env_left(self, i):
        tmp = torch.tensordot(self.env[i], self.mps[i], [[0], [0]])
        tmp = torch.tensordot(tmp, self.mpo[i], [[0, 3], [0, 2]])
        self.env[i + 1] = torch.tensordot(tmp, self.mps[i].conj(), [[0, 3], [0, 2]])

    def _update_env_right(self, i):
        tmp = torch.tensordot(self.env[i + 3], self.mps[i + 1], [[0], [1]])
        tmp = torch.tensordot(tmp, self.mpo[i + 1], [[0, 3], [1, 2]])
        self.env[i + 2] = torch.tensordot(tmp, self.mps[i + 1].conj(), [[0, 3], [1, 2]])
 

def random_ttn(N, d, D, physical_indices = None):
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

def get_ising_mpo(n: int, J = -1.0, gx = 0, gz = 0, physical_indices = None) -> MPO:
    
    """
    Build and return an MPO object for the 1D transverse field Ising model,
    but now using Node/Tensor/Index structure.
    """
    
    I = torch.eye(2, dtype=torch.float64)
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.float64)

    D = 3
    d = 2

    # Define shared physical indices for all sites
    if physical_indices == None:
        phys_in = [Index(d, name=f"phys_{i}") for i in range(n)]
        phys_out = [Index(d, name=f"phys_{i}'") for i in range(n)]
    else:
        phys_in = physical_indices
        phys_out = list(map(lambda idx : idx.prime(), physical_indices))
    
    # Bond indices
    bonds = [Index(D, name=f"bond_{i}") for i in range(1, n)]

    nodes = []

    # Left boundary
    Dl, Dr = 1, D
    bond_right = bonds[0]
    indices = [Index(Dl, name="bond_0"), bond_right, phys_in[0], phys_out[0]]
    W_left = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
    W_left[0, 0] = gx * x + gz * z
    W_left[0, 1] = J * z
    W_left[0, 2] = I
    tensor_left = Tensor(indices, W_left)
    node_left = Node(tensor_left)
    nodes.append(node_left)

    # Bulk
    for i in range(1, n - 1):
        Dl, Dr = D, D
        bond_left = bonds[i - 1]
        bond_right = bonds[i]
        indices = [bond_left, bond_right, phys_in[i], phys_out[i]]
        W = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
        W[0, 0] = I
        W[1, 0] = z
        W[2, 0] = gx * x + gz * z
        W[2, 1] = J * z
        W[2, 2] = I
        tensor_bulk = Tensor(indices, W)
        node_bulk = Node(tensor_bulk)
        nodes.append(node_bulk)

    # Right boundary
    Dl, Dr = D, 1
    bond_left = bonds[-1]
    indices = [bond_left, Index(Dr, name=f"bond_{n}"), phys_in[n - 1], phys_out[n - 1]]
    W_right = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
    W_right[0, 0] = I
    W_right[1, 0] = z
    W_right[2, 0] = gx * x + gz * z
    tensor_right = Tensor(indices, W_right)
    node_right = Node(tensor_right)
    nodes.append(node_right)

    # Return MPO object (assuming now you have MPO that takes nodes)
    
    return MPO(n=n, d=d, nodes=nodes)

def product_state_ttn(product_state, d, physical_indices = None):
   
    N = len(product_state) 
   
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
        bond = Index(1, name=f"bond_{node_id}")
        data = torch.zeros(phys[i].dim, phys[i+1].dim, 1, dtype = torch.float64)
        data[product_state[i], product_state[i+1], 0] = 1
        tensor = Tensor([phys[i], phys[i + 1], bond], data = data)
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
                tensor = Tensor([left.tensor.indices[-1], right.tensor.indices[-1]], data = torch.ones(1, 1, dtype = torch.float64))
            else:
                bond = Index(1, name=f"bond_{node_id}")
                tensor = Tensor([left.tensor.indices[-1], right.tensor.indices[-1], bond], data = torch.ones(1, 1, 1, dtype = torch.float64))

            parent = Node(tensor, parents=[], children=[left, right], name=node_id, layer=current_layer)
            left.parents.append(parent)
            right.parents.append(parent)

            next_nodes.append(parent)
            nodes.append(parent)
            node_id += 1

        current_nodes = next_nodes

    return TTN(nodes=nodes)
  
def get_two_site_mpo(n: int, site1, site2, op, physical_indices = None) -> MPO:
    
    """
    Build and return an MPO object for the 1D transverse field Ising model,
    but now using Node/Tensor/Index structure.
    """
    
    I = torch.eye(2, dtype=torch.float64)

    d = 2

    # Define shared physical indices for all sites
    if physical_indices == None:
        phys_in = [Index(d, name=f"phys_{i}") for i in range(n)]
        phys_out = [Index(d, name=f"phys_{i}'") for i in range(n)]
    else:
        phys_in = physical_indices
        phys_out = list(map(lambda idx : idx.prime(), physical_indices))
    
    # Bond indices
    bonds = [Index(1, name=f"bond_{i}") for i in range(1, n)]

    nodes = []

    # Left boundary
    Dl, Dr = 1, 1
    bond_right = bonds[0]
    indices = [Index(Dl, name="bond_0"), bond_right, phys_in[0], phys_out[0]]
    W_left = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
    W_left[0, 0] = op if site1 == 0 else I
    tensor_left = Tensor(indices, W_left)
    node_left = Node(tensor_left)
    nodes.append(node_left)

    # Bulk
    for i in range(1, n - 1):
        Dl, Dr = 1, 1
        bond_left = bonds[i - 1]
        bond_right = bonds[i]
        indices = [bond_left, bond_right, phys_in[i], phys_out[i]]
        W = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
        W[0, 0] = op if site1 == i or site2 == i else I
        tensor_bulk = Tensor(indices, W)
        node_bulk = Node(tensor_bulk)
        nodes.append(node_bulk)

    # Right boundary
    Dl, Dr = 1, 1
    bond_left = bonds[-1]
    indices = [bond_left, Index(Dr, name=f"bond_{n}"), phys_in[n - 1], phys_out[n - 1]]
    W_right = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
    W_right[0, 0] = op if site2 == n-1 else I
    tensor_right = Tensor(indices, W_right)
    node_right = Node(tensor_right)
    nodes.append(node_right)

    # Return MPO object (assuming now you have MPO that takes nodes)
    
    return MPO(n=n, d=d, nodes=nodes)    

def get_single_site_mpo(n: int, site1, op, physical_indices = None) -> MPO:
    
    """
    Build and return an MPO object for the 1D transverse field Ising model,
    but now using Node/Tensor/Index structure.
    """
    
    I = torch.eye(2, dtype=torch.float64)

    d = 2

    # Define shared physical indices for all sites
    if physical_indices == None:
        phys_in = [Index(d, name=f"phys_{i}") for i in range(n)]
        phys_out = [Index(d, name=f"phys_{i}'") for i in range(n)]
    else:
        phys_in = physical_indices
        phys_out = list(map(lambda idx : idx.prime(), physical_indices))
    
    # Bond indices
    bonds = [Index(1, name=f"bond_{i}") for i in range(1, n)]

    nodes = []

    # Left boundary
    Dl, Dr = 1, 1
    bond_right = bonds[0]
    indices = [Index(Dl, name="bond_0"), bond_right, phys_in[0], phys_out[0]]
    W_left = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
    W_left[0, 0] = op if site1 == 0 else I
    tensor_left = Tensor(indices, W_left)
    node_left = Node(tensor_left)
    nodes.append(node_left)

    # Bulk
    for i in range(1, n - 1):
        Dl, Dr = 1, 1
        bond_left = bonds[i - 1]
        bond_right = bonds[i]
        indices = [bond_left, bond_right, phys_in[i], phys_out[i]]
        W = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
        W[0, 0] = op if site1 == i else I
        tensor_bulk = Tensor(indices, W)
        node_bulk = Node(tensor_bulk)
        nodes.append(node_bulk)

    # Right boundary
    Dl, Dr = 1, 1
    bond_left = bonds[-1]
    indices = [bond_left, Index(Dr, name=f"bond_{n}"), phys_in[n - 1], phys_out[n - 1]]
    W_right = torch.zeros(Dl, Dr, d, d, dtype=torch.float64)
    W_right[0, 0] = op if site1 == n-1 else I
    tensor_right = Tensor(indices, W_right)
    node_right = Node(tensor_right)
    nodes.append(node_right)

    # Return MPO object (assuming now you have MPO that takes nodes)
    
    return MPO(n=n, d=d, nodes=nodes)    


if __name__ == "__main__":

    depth = 3
    N = 2**depth
    d = 2
    D = 1

    # t1 = random_ttn(N, d, D).canonicalize(normalize = True)

    product_state = [0]*N
    t1 = product_state_ttn(product_state, d)

    # t1.visualize()

    # pi = t1.get_physical_indices()
    # t2 = random_binary_ttn(N, d, D, pi)

    mpo = get_ising_mpo(N, physical_indices = t1.get_physical_indices())

    print(t1.expectation_value(mpo))
