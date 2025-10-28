import torch
import numpy as np
from typing import List, Optional, Callable
import torch.optim as optim
from datetime import datetime
from torch.utils.checkpoint import checkpoint

# torch.autograd.set_detect_anomaly(True)

def get_linkdims(tensors):
        
        linkdims = []
        n = len(tensors)
        for i in range(n-1):
            linkdims.append(tensors[i].shape[1])
            
        return [1] + linkdims + [1]

class MPS(torch.nn.Module):
    
    def __init__(self, n: int, d: int, linkdims: List[int] = [], requires_grad: bool = False, tensors: List[torch.Tensor] = None):
        
        """
        Matrix Product State (MPS) initialization with random tensors.

        Args:
            n (int): Number of sites.
            d (int): Physical dimension.
            linkdims (List[int]): Bond dimensions of length n+1.
            requires_grad (bool): Whether the tensors require gradients.
        """
        
        super().__init__()
        self.n = n
        self.d = d
        self.requires_grad = requires_grad
        if tensors == None:
            self.linkdims = linkdims
            self.tensors = self._initialize_random_mps()
        else:
            self.tensors = tensors
            self.linkdims = get_linkdims(tensors)

    def _initialize_random_mps(self) -> List[torch.Tensor]:
    
        tensors = []        
        for i in range(self.n):
            tensor = torch.rand(self.linkdims[i], self.linkdims[i + 1], self.d, dtype=torch.float64, requires_grad = self.requires_grad)
            tensors.append(tensor)
        
        return tensors

    def __getitem__(self, idx):
                
        return self.tensors[idx]

    def __setitem__(self, idx, value):
        
        self.tensors[idx].data = value.data

    def __rmul__(self, scalar):
        
        # Multiply the MPS by the scalar
        
        self.tensors[0] *= scalar
        
        return self

    def normalize(self, requires_grad = False):
        
        """Left-canonicalize the MPS using QR decomposition and normalize it."""
        
        if requires_grad:
        
            mps_copy = self.tensors
            res = []
            
            Dl, Dr, d = mps_copy[0].shape
            tmp = mps_copy[0].permute(0, 2, 1).reshape(Dl * d, Dr)
            q, r = torch.linalg.qr(tmp)
            if self.n > 1:
                right = torch.tensordot(r, mps_copy[1], dims=([1], [0]))
            new_dim = q.shape[-1]
            res.append(q.reshape(Dl, d, new_dim).permute(0, 2, 1))        
            
            for i in range(1, self.n):
                
                Dl, Dr, d = right.shape
                tmp = right.permute(0, 2, 1).reshape(Dl * d, Dr)
                q, r = torch.linalg.qr(tmp)

                if i == self.n - 1:
                    r = r / torch.norm(r)
                    res.append(torch.tensordot(q, r, [[1], [0]]).reshape(Dl, d, Dr).permute(0, 2, 1)) 
                else:
                    right = torch.tensordot(r, mps_copy[i + 1], dims=([1], [0]))
                    new_dim = q.shape[-1]
                    res.append(q.reshape(Dl, d, new_dim).permute(0, 2, 1)) 
            
            linkdims = get_linkdims(res)
                    
            return MPS(self.n, d, linkdims, tensors = res, requires_grad = requires_grad)
        
        else:
            
            with torch.no_grad():
                
                mps_copy = self.tensors
                res = []
                
                Dl, Dr, d = mps_copy[0].shape
                tmp = mps_copy[0].permute(0, 2, 1).reshape(Dl * d, Dr)
                q, r = torch.linalg.qr(tmp)
                if self.n > 1:
                    right = torch.tensordot(r, mps_copy[1], dims=([1], [0]))
                new_dim = q.shape[-1]
                res.append(q.reshape(Dl, d, new_dim).permute(0, 2, 1))        
                
                for i in range(1, self.n):
                    
                    Dl, Dr, d = right.shape
                    tmp = right.permute(0, 2, 1).reshape(Dl * d, Dr)
                    q, r = torch.linalg.qr(tmp)

                    if i == self.n - 1:
                        r = r / torch.norm(r)
                        res.append(torch.tensordot(q, r, [[1], [0]]).reshape(Dl, d, Dr).permute(0, 2, 1)) 
                    else:
                        right = torch.tensordot(r, mps_copy[i + 1], dims=([1], [0]))
                        new_dim = q.shape[-1]
                        res.append(q.reshape(Dl, d, new_dim).permute(0, 2, 1)) 
                
                linkdims = get_linkdims(res)
                        
                return MPS(self.n, d, linkdims, tensors = res, requires_grad = requires_grad)

    def __repr__(self):
        
        return f"MPS(n={self.n}, d={self.d}, bond_dims={self.linkdims})"
    
    def compress(self, tol = 1e-14, maxD = np.inf, requires_grad = False):
        
        n = self.n
        d = self.d
        
        res = [self.tensors[i].clone() for i in range(n)]
        
        # Left canonical form up to penultimate site
        for i in range(self.n-1):
            
            dims = res[i].shape
            tmp = res[i].permute(0, 2, 1).reshape(dims[0]*dims[2], dims[1])
            q, r = torch.linalg.qr(tmp)
            res[i+1] = torch.tensordot(r, res[i+1], [[1], [0]])
            res[i] = q.reshape(dims[0], dims[2], q.shape[-1]).permute(0, 2, 1)

        # Right canonical form with truncation
        for i in reversed(range(1, self.n)):
            
            dims = res[i].shape
            tmp = res[i].reshape(dims[0], dims[1]*dims[2])
            u, s, vh = torch.linalg.svd(tmp, full_matrices=False)
            D = min((torch.abs(s) > tol).sum().item(), maxD)
            u, s, vh = u[:, :D], torch.diag(s[:D]), vh[:D, :].reshape(D, dims[1], dims[2])
            res[i] = vh
            res[i-1] = torch.tensordot(res[i-1], torch.tensordot(u, s, [[1], [0]]), [[1], [0]]).permute(0, 2, 1)
        
        linkdims = get_linkdims(res)
                                           
        return MPS(self.n, d, linkdims, tensors = res, requires_grad = requires_grad)

    def inner(self, other: 'MPS') -> torch.Tensor:
        
        """
        Calculate <self|other> inner product assuming same length.
        """
        
        n = self.n

        dims1 = self.tensors[0].shape
        dims2 = other.tensors[0].shape

        res = torch.tensordot(self.tensors[0].conj().reshape(dims1[1], dims1[2]),
                              other.tensors[0].reshape(dims2[1], dims2[2]), [[1], [1]])

        for i in range(1, n):
            res = torch.tensordot(res, other.tensors[i], [[1], [0]])
            res = torch.tensordot(res, self.tensors[i].conj(), [[0, 2], [0, 2]]).permute(1, 0)

        return res

    def norm(self) -> torch.Tensor:
        
        """
        Return the norm squared of the MPS.
        """
        
        return torch.sqrt(self.inner(self))
    
    def contract_with_mpo(self, mpo, tol = 1e-14, maxD = np.inf, compress = False, requires_grad = False):
        
        n = self.n
        d = self.d
        res = []

        E = []
        tmp1 = torch.tensor([[[[1.0]]]], dtype=torch.float64)

        for i in range(n - 1):
            tmp2 = torch.tensordot(tmp1, self.tensors[i].conj(), [[0], [0]])
            tmp3 = torch.tensordot(tmp2, mpo.tensors[i].conj(), [[0, 4], [0, 2]])
            tmp4 = torch.tensordot(tmp3, mpo.tensors[i], [[0, 4], [0, 2]])
            tmp1 = torch.tensordot(tmp4, self.tensors[i], [[0, 4], [0, 2]])
            E.append(tmp1)

        dims = self.tensors[-1].shape
        mps_conj = self.tensors[-1].conj().reshape(dims[0], dims[2])
        dims = mpo.tensors[-1].shape
        mpo_conj = mpo.tensors[-1].conj().reshape(dims[0], dims[2], dims[3])
        lower_right_part = torch.tensordot(mps_conj, mpo_conj, [[1], [1]])

        rho = torch.tensordot(E[-1], lower_right_part, [[0, 1], [0, 1]])
        rho = torch.tensordot(rho, lower_right_part.conj(), [[1, 0], [0, 1]])

        evals, U = torch.linalg.eigh(rho)
        if compress:
            D = min((torch.abs(evals) > tol).sum().item(), maxD)
        else:
            D = (torch.abs(evals) > 1e-14).sum().item()

        U = torch.flip(U, dims=[1])[:, :D].T.conj()
        dims = U.shape
        res.insert(0, U.reshape(dims[0], 1, dims[1]))

        C = torch.tensordot(lower_right_part, U.conj(), [[2], [1]])

        for i in range(n - 2, 0, -1):
            
            lower_middle_part = torch.tensordot(self.tensors[i].conj(), mpo.tensors[i].conj(), [[2], [2]])
            lower_part = torch.tensordot(lower_middle_part, C, [[1, 3], [0, 1]])
            rho = torch.tensordot(E[i - 1], lower_part, [[0, 1], [0, 1]])
            rho = torch.tensordot(rho, lower_part.conj(), [[0, 1], [1, 0]])

            dims = rho.shape
            rho = rho.reshape(dims[0] * dims[1], dims[2] * dims[3])
            evals, U = torch.linalg.eigh(rho)

            if compress:
                D = min((torch.abs(evals) > tol).sum().item(), maxD)
            else:
                D = (torch.abs(evals) > 1e-14).sum().item()

            U = torch.flip(U, dims=[1])[:, :D].T.conj()
            U = U.reshape(U.shape[0], dims[2], dims[3]).permute(0, 2, 1)

            res.insert(0, U)
            C = torch.tensordot(lower_part, U.conj(), [[3, 2], [1, 2]])

        dims = mpo.tensors[0].shape
        mpo_conj = mpo.tensors[0].conj().reshape(dims[1], dims[2], dims[3])
        left_part = torch.tensordot(self.tensors[0].conj(), mpo_conj, [[2], [1]])
        mps0 = torch.tensordot(left_part, C, [[1, 2], [0, 1]])
        res.insert(0, mps0.permute(0, 2, 1))
        
        return MPS(n, d, linkdims = get_linkdims(res), tensors = res, requires_grad = requires_grad)
    
    def get_expectation_value(self, mpo, requires_grad = False):
                
        return self.inner(self.contract_with_mpo(mpo, requires_grad = requires_grad))
        
class MPO:
    
    def __init__(self, n: int, d: int, builder_fn: Optional[Callable[[int], List[torch.Tensor]]] = None, tensors: Optional[List[torch.Tensor]] = None):
        
        """
        Matrix Product Operator (MPO) wrapper class.

        Args:
            n (int): Number of sites.
            builder_fn (Callable[[int], List[torch.Tensor]], optional): Function to build MPO tensors.
            tensors (List[torch.Tensor], optional): Predefined list of MPO tensors.
        """
        
        self.n = n
        self.d = d
        if tensors is not None:
            self.tensors = tensors
        elif builder_fn is not None:
            self.tensors = builder_fn(n)
        else:
            raise ValueError("Either builder_fn or tensors must be provided.")
        self.linkdims = get_linkdims(tensors)

    def get_tensors(self) -> List[torch.Tensor]:
        
        return self.tensors

    def __repr__(self):
        
        return f"MPO(n={self.n}, tensors={[t.shape for t in self.tensors]})"
    
    def __getitem__(self, idx):
        
        return self.tensors[idx]
    
    def __setitem__(self, idx, value):
        
        self.tensors[idx] = value
        
    def to_matrix(self):
        
        n = self.n
        d = self.d
        tmp = torch.tensor([1.0], dtype=torch.float64)

        res = torch.tensordot(self.tensors[-1].to(dtype=torch.float64), tmp, dims=([1], [0]))
        for i in reversed(range(n - 1)):
            res = torch.tensordot(self.tensors[i].to(dtype=torch.float64), res, dims=([1], [0]))

        res = torch.tensordot(tmp, res, dims=([0], [0]))

        # Permute to move physical indices to outermost positions
        perm = torch.cat([torch.arange(1, 2 * n, 2), torch.arange(0, 2 * n, 2)]).tolist()
        res = res.permute(*perm).reshape(d ** n, d ** n)

        return res
    
class DMRG:
    
    def __init__(self, mpo : 'MPO', mps : 'MPS', sweeps = 10, etol = 1e-14, maxD = np.inf, tol = 1e-14, compress = True, verbose = False):
        
        self.mpo = mpo
        self.mps = mps 
        self.sweeps = sweeps
        self.etol = etol
        self.maxD = maxD
        self.tol = tol
        self.compress = compress
        self.verbose = verbose
        self.n = self.mps.n
        self.d = self.mps.d
        self.env = self._initialize_env()

    def _initialize_env(self):
        
        env = [torch.tensor([[[1.0]]], dtype=torch.float64)] + [None for _ in range(self.n)] + [torch.tensor([[[1.0]]], dtype=torch.float64)]
        for i in reversed(range(self.n)):
            tmp = torch.tensordot(env[i + 2], self.mps[i], [[0], [1]])
            tmp = torch.tensordot(tmp, self.mpo[i], [[0, 3], [1, 2]])
            env[i + 1] = torch.tensordot(tmp, self.mps[i].conj(), [[0, 3], [1, 2]])
        return env

    def run(self):
        
        # E_curr = 1e-16
        E_curr = (self.mps.get_expectation_value(self.mpo)/self.mps.inner(self.mps)).item()
        print(f'Initial Energy: {E_curr}')

        for sweep in range(self.sweeps):
            
            E = self._sweep_lr(sweep)
            E = self._sweep_rl(sweep)
            frac_change = abs((E - E_curr) / E_curr)
            E_curr = E

            print(f"Sweep {sweep}, E = {E}")
            if frac_change < self.etol:
                print("Energy accuracy reached.")
                return E_curr, MPS(self.n, self.d, get_linkdims(self.mps.tensors), requires_grad = True, tensors = self.mps.tensors)

        print("Maximum sweeps reached before reaching desired energy accuracy.")
        return E_curr, MPS(self.n, self.d, get_linkdims(self.mps.tensors), requires_grad = True, tensors = self.mps.tensors)

    def _sweep_lr(self, sweep):
        for i in range(self.n - 1):
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
        for i in reversed(range(self.n - 1)):
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
             
def get_ising_mpo(n: int, J=1.0, gx=1.0, gz=1.0) -> MPO:
    
    """
    Build and return an MPO object for the 1D transverse field Ising model.
    """
    
    I = torch.eye(2, dtype=torch.float64)
    x = torch.tensor([[0., 1.], [1., 0.]], dtype=torch.float64)
    z = torch.tensor([[1., 0.], [0., -1.]], dtype=torch.float64)

    D = 3
    d = 2

    W = torch.zeros(D, D, d, d, dtype=torch.float64)
    W[0, 0] = I
    W[1, 0] = z
    W[2, 0] = gx * x + gz * z
    W[2, 1] = J * z
    W[2, 2] = I

    W_left = torch.zeros(1, D, d, d, dtype=torch.float64)
    W_left[0, 0] = gx * x + gz * z
    W_left[0, 1] = J * z
    W_left[0, 2] = I

    W_right = torch.zeros(D, 1, d, d, dtype=torch.float64)
    W_right[0, 0] = I
    W_right[1, 0] = z
    W_right[2, 0] = gx * x + gz * z

    mpo_tensors = [W_left] + [W.clone() for _ in range(n - 2)] + [W_right]
    
    return MPO(n = n, d = 2, tensors=mpo_tensors)

n = 10
d = 2
n_epochs = 100
tol = 1e-11  # Convergence tolerance for loss
ising_mpo = get_ising_mpo(n, J=1.0, gx=0.2, gz=0.1)

linkdims = [1] + [8 for _ in range(n-1)] + [1]
mps = MPS(n = n, d = 2, linkdims = linkdims, requires_grad = True)

def loss_fn():
    
    energy = mps.get_expectation_value(ising_mpo)
    norm = mps.inner(mps)
    loss = energy / norm
    
    return loss

optimizer = torch.optim.LBFGS(mps.tensors)

prev_loss = float('inf')

for epoch in range(n_epochs):
    
    def closure():
    
        optimizer.zero_grad()
        loss = loss_fn()
        loss.backward()
        
        return loss

    loss = optimizer.step(closure)
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch+1}: Energy = {loss.item()}, Ranks = {get_linkdims(mps.tensors)}")

    # Early stopping condition
    if abs(prev_loss - loss.item()) < tol and epoch > 50:
        print(f"Converged at epoch {epoch+1}: Δloss = {abs(prev_loss - loss.item())} < {tol}\n")
        break

    prev_loss = loss.item()

mps = mps.normalize(requires_grad = False) 
print('Trained mps expectation value with ising mpo:', mps.get_expectation_value(ising_mpo).item())

mps = MPS(n, d, linkdims)

dmrg = DMRG(ising_mpo, mps, etol = 1e-8, tol = 1e-8, maxD = 40, verbose = False)
E, gs = dmrg.run()
    
print('DMRG:', E, ',', gs) 

H = ising_mpo.to_matrix()
evals, evecs = torch.linalg.eigh(H)
print('ED:  ', evals[0].item())
