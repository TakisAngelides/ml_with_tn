import torch
import numpy as np
from typing import List, Optional, Callable

class MPS:
    
    def __init__(self, N: int, d: int, linkdims: List[int] = None, requires_grad: bool = False, tensors: List[torch.Tensor] = None):
        
        """
        Matrix Product State (MPS) initialization with random tensors.

        Args:
            N (int): Number of sites.
            d (int): Physical dimension.
            linkdims (List[int]): Bond dimensions of length N+1.
            requires_grad (bool): Whether the tensors require gradients.
        """
        
        self.N = N
        self.d = d
        self.requires_grad = requires_grad
        if tensors == None:
            if linkdims == None:
                self.linkdims = list(np.ones(N+1, dtype = np.int64))
            else:
                self.linkdims = linkdims
            self.tensors = self._initialize_random_mps()
            self.normalize()
        else:
            self.tensors = tensors
            self.linkdims = get_linkdims(tensors)

    def _initialize_random_mps(self) -> List[torch.Tensor]:
        
        mps = []
        for i in range(self.N):
            tensor = torch.rand(self.linkdims[i], self.linkdims[i + 1], self.d, dtype=torch.float64, requires_grad=self.requires_grad)
            mps.append(tensor)
        
        return mps

    def __getitem__(self, idx):
                
        return self.tensors[idx]

    def __setitem__(self, idx, value):
        
        self.tensors[idx] = value

    def __rmul__(self, scalar):
        
        # Multiply the MPS by the scalar
        
        self.tensors[0] *= scalar
        
        return self

    def normalize(self):
        
        """Left-canonicalize the MPS using QR decomposition and normalize it."""
        
        for i in range(self.N):
            A = self.tensors[i]
            Dl, Dr, d = A.shape
            tmp = A.permute(0, 2, 1).reshape(Dl * d, Dr)
            q, r = torch.linalg.qr(tmp)
            if i < self.N - 1:
                self.tensors[i + 1] = torch.tensordot(r, self.tensors[i + 1], dims=([1], [0]))
            new_dim = q.shape[-1]
            self.tensors[i] = q.reshape(Dl, d, new_dim).permute(0, 2, 1)
            
        self.linkdims = get_linkdims(self.tensors)
            
        return self

    def __repr__(self):
        
        return f"MPS(N={self.N}, d={self.d}, bond_dims={self.linkdims})"
    
    def compress(self, tol = 1e-14, maxD = np.inf):
        
        # Left canonical form up to penultimate site
        for i in range(self.N-1):
            dims = self[i].shape
            tmp = self[i].permute(0, 2, 1).reshape(dims[0]*dims[2], dims[1])
            q, r = torch.linalg.qr(tmp)
            self[i+1] = torch.tensordot(r, self[i+1], [[1], [0]])
            self[i] = q.reshape(dims[0], dims[2], q.shape[-1]).permute(0, 2, 1)

        # Right canonical form with truncation
        for i in reversed(range(1, self.N)):
            dims = self[i].shape
            tmp = self[i].reshape(dims[0], dims[1]*dims[2])
            u, s, vh = torch.linalg.svd(tmp, full_matrices=False)
            D = min((torch.abs(s) > tol).sum().item(), maxD)
            u, s, vh = u[:, :D], torch.diag(s[:D]), vh[:D, :].reshape(D, dims[1], dims[2])
            self[i] = vh
            self.linkdims[i] = vh.shape[0]
            self[i-1] = torch.tensordot(self[i-1], torch.tensordot(u, s, [[1], [0]]), [[1], [0]]).permute(0, 2, 1)
                    
        return self

    def inner(self, other: 'MPS') -> torch.Tensor:
        
        """
        Calculate <self|other> inner product assuming same length.
        """
        
        N = self.N

        dims1 = self.tensors[0].shape
        dims2 = other.tensors[0].shape

        res = torch.tensordot(self.tensors[0].conj().reshape(dims1[1], dims1[2]),
                              other.tensors[0].reshape(dims2[1], dims2[2]), [[1], [1]])

        for i in range(1, N):
            res = torch.tensordot(res, other.tensors[i], [[1], [0]])
            res = torch.tensordot(res, self.tensors[i].conj(), [[0, 2], [0, 2]]).permute(1, 0)

        return res

    def norm(self) -> torch.Tensor:
        
        """
        Return the norm squared of the MPS.
        """
        
        return torch.sqrt(self.inner(self))

    def set_mixed_canonical_form(self, last_left: int):
        
        """
        Put the MPS in mixed canonical form with orthogonality center at last_left.
        """
        
        N = self.N
        mps = self.tensors.copy()

        # Left canonical for sites up to last_left
        for i in range(min(last_left + 1, N-1)):
            dims = mps[i].shape
            tmp = mps[i].permute(0, 2, 1).reshape(dims[0]*dims[2], dims[1])
            q, r = torch.linalg.qr(tmp)
            mps[i] = q.reshape(dims[0], dims[2], q.shape[-1]).permute(0, 2, 1)
            mps[i+1] = torch.tensordot(r, mps[i+1], [[1], [0]])

        # Right canonical for sites from N-1 down to last_left+1
        for i in range(N-1, max(last_left+1, 0), -1):
            
            dims = mps[i].shape
            tmp = mps[i].reshape(dims[0], dims[1]*dims[2])
            u, s, v = torch.linalg.svd(tmp, full_matrices=False)
            mps[i] = v.reshape(v.shape[0], dims[1], dims[2])
            mps[i-1] = torch.tensordot(mps[i-1], torch.tensordot(u, torch.diag(s), [[1], [0]]), [[1], [0]]).permute(0, 2, 1)

        self.tensors = mps
        
        return self

    @staticmethod
    def _get_site_canonical_form(site: torch.Tensor, tol = 1e-14) -> str:
        
        """
        Check canonical form type of a single site tensor.

        Returns:
            "L" if left-canonical,
            "R" if right-canonical,
            "M" if mixed canonical (both),
            "N" if none.
        """
        
        left = torch.tensordot(site.conj(), site, [[0, 2], [0, 2]])
        right = torch.tensordot(site.conj(), site, [[1, 2], [1, 2]])

        left_ok = torch.dist(torch.eye(left.size(0)), left).item() < tol
        right_ok = torch.dist(torch.eye(right.size(0)), right).item() < tol

        if left_ok and not right_ok:
            return "L"
        elif right_ok and not left_ok:
            return "R"
        elif right_ok and left_ok:
            return "M"
        else:
            return "N"

    def get_canonical_form(self) -> list[str]:
        
        """
        Return list of canonical form indicators ("L", "R", "M", "N") for each site.
        """
        
        return [self._get_site_canonical_form(site) for site in self.tensors]
    
    def contract_with_mpo(self, mpo, tol = 1e-14, maxD = np.inf, compress = False):
        
        N = self.N
        d = self.d
        res = []

        E = []
        tmp1 = torch.tensor([[[[1.0]]]], dtype=torch.float64)

        for i in range(N - 1):
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

        for i in range(N - 2, 0, -1):
            
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
        
        return MPS(N, d, linkdims = get_linkdims(res), tensors = res)
    
    def get_expectation_value(self, mpo):
            
        return self.inner(self.contract_with_mpo(mpo))
    
    def get_entanglement_entropy(self, link):
        
        mps = self.normalize_mps(self.mps)
        mps = self.set_mixed_canonical_form(mps, link-1)
        tmp = torch.tensordot(mps[link-1], mps[link], [[1], [0]])
        dims = tmp.shape
        tmp = tmp.reshape(dims[0]*dims[1], dims[2]*dims[3])
        _, s, _ = torch.linalg.svd(tmp, full_matrices=False)
        entropy = 0
        for element in s:
            if torch.abs(element) < self.tol:
                continue
            entropy += -element**2 * torch.log2(element**2)
            
        return entropy.item()

    def get_single_site_op_mpo(self, N, d, site, op):
        
        I = torch.eye(d, dtype=torch.float64)
        mpo = []
        for i in range(N):
            site_op = op if i == site else I
            mpo_tensor = site_op.reshape(1, 1, d, d)
            mpo.append(mpo_tensor)
        
        return mpo

    def get_single_site_op_sum_mpo(self, N, d, op):
        
        I = torch.eye(d, dtype=torch.float64)
        mpo = []

        # First site
        W0 = torch.stack([I, op], dim=0).unsqueeze(0)
        mpo.append(W0)

        # Middle sites
        for _ in range(1, N - 1):
            W = torch.zeros((2, 2, d, d), dtype=torch.float64)
            W[0, 0] = I
            W[1, 0] = op
            W[1, 1] = I
            mpo.append(W)

        # Last site
        Wn = torch.stack([I, op], dim=0).unsqueeze(1)
        mpo.append(Wn)

        return mpo

    def to_list(self):
        
        N, d = self.N, self.d
                     
        res = self.tensors[0]
        for i in range(1, N):
            res = torch.tensordot(res, self.tensors[i], [[-2], [0]])
            
        return res.reshape(d**N).tolist()

class MPO:
    
    def __init__(self, N: int, d: int, builder_fn: Optional[Callable[[int], List[torch.Tensor]]] = None, tensors: Optional[List[torch.Tensor]] = None):
        
        """
        Matrix Product Operator (MPO) wrapper class.

        Args:
            N (int): Number of sites.
            builder_fn (Callable[[int], List[torch.Tensor]], optional): Function to build MPO tensors.
            tensors (List[torch.Tensor], optional): Predefined list of MPO tensors.
        """
        
        self.N = N
        self.d = d
        if tensors is not None:
            self.tensors = tensors
        elif builder_fn is not None:
            self.tensors = builder_fn(N)
        else:
            raise ValueError("Either builder_fn or tensors must be provided.")
        self.linkdims = get_linkdims(tensors)

    def get_tensors(self) -> List[torch.Tensor]:
        
        return self.tensors

    def __repr__(self):
        
        return f"MPO(N={self.N}, tensors={[t.shape for t in self.tensors]})"
    
    def __getitem__(self, idx):
        
        return self.tensors[idx]
    
    def __setitem__(self, idx, value):
        
        self.tensors[idx] = value
    
    def to_matrix(self):
        
        N = self.N
        d = self.d
        tmp = torch.tensor([1.0], dtype=torch.float64)

        res = torch.tensordot(self.tensors[-1].to(dtype=torch.float64), tmp, dims=([1], [0]))
        for i in reversed(range(N - 1)):
            res = torch.tensordot(self.tensors[i].to(dtype=torch.float64), res, dims=([1], [0]))

        res = torch.tensordot(tmp, res, dims=([0], [0]))

        # Permute to move physical indices to outermost positions
        perm = torch.cat([torch.arange(1, 2 * N, 2), torch.arange(0, 2 * N, 2)]).tolist()
        res = res.permute(*perm).reshape(d ** N, d ** N)

        return res
    
    def compress(self, tol = 1e-14, maxD = np.inf):
        
        N = self.N
        res = []

        # Left canonicalization via QR
        for i in range(N - 1):
            
            Dl, Dr, d1, d2 = self[i].shape
            M = self[i].permute(0, 2, 3, 1).reshape(Dl * d1 * d1, Dr)
            Q, R = torch.linalg.qr(M)
            newD = Q.shape[1]
            Qt = Q.reshape(Dl, d1, d1, newD).permute(0, 3, 1, 2)
            res.append(Qt)
            Dl_next, Dr_next, d3, d4 = self[i + 1].shape
            self[i + 1] = torch.tensordot(R, self[i + 1].reshape(Dr, -1), [[1], [0]]).reshape(newD, Dr_next, d3, d4)
        
        res.append(self[-1])

        # Right to left SVD truncation
        for i in reversed(range(1, N)):
            
            Dl, Dr, d1, d2 = res[i].shape
            M = res[i].reshape(Dl, Dr * d1 * d2)
            U, S, Vh = torch.linalg.svd(M, full_matrices=False)
            keep = min(int((S.abs() > tol).sum().item()), maxD)
            U, S, Vh = U[:, :keep], S[:keep], Vh[:keep, :]
            res[i] = Vh.reshape(keep, Dr, d1, d2)
            B = U @ torch.diag(S)
            C = torch.tensordot(res[i - 1], B, [[1], [0]])
            res[i - 1] = C.permute(0, 3, 1, 2)

        return MPO(N = N, d = 2, tensors = res)
    
class DMRG:
    
    def __init__(self, mpo : MPO, mps : MPS, sweeps = 10, etol = 1e-14, maxD = np.inf, tol = 1e-14, compress = True, verbose = False):
        
        self.mpo = mpo
        self.mps = mps 
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


def get_ising_mpo(N: int, J=1.0, gx=1.0, gz=1.0) -> MPO:
    
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

    mpo_tensors = [W_left] + [W.clone() for _ in range(N - 2)] + [W_right]
    
    return MPO(N = N, d = 2, tensors=mpo_tensors)

def get_linkdims(tensors):
        
        linkdims = []
        N = len(tensors)
        for i in range(N-1):
            linkdims.append(tensors[i].shape[1])
            
        return [1] + linkdims + [1]

def add_mps(mps1, mps2, tol = 1e-14, maxD = np.inf, compress = False, normalize = False):
        
        N = mps1.N
        d = mps1.d
        D1, D2 = mps1.linkdims, mps2.linkdims
                
        res = []
        for i in range(N):
            
            if i == 0:
                
                a, b = D1[i + 1], D2[i + 1]
                tmp = torch.zeros(1, a + b, d)
                tmp[:, :a, :] = mps1[i]
                tmp[:, a:, :] = mps2[i]
                res.append(tmp)
                
            elif i == N - 1:
                
                a, b = D1[i], D2[i]
                tmp = torch.zeros(a + b, 1, d)
                tmp[:a, :, :] = mps1[i]
                tmp[a:, :, :] = mps2[i]
                res.append(tmp)
                
            else:
                
                a_left, b_left = D1[i], D2[i]
                a_right, b_right = D1[i + 1], D2[i + 1]
                tmp = torch.zeros(a_left + b_left, a_right + b_right, d)
                tmp[:a_left, :a_right, :] = mps1[i]
                tmp[a_left:, a_right:, :] = mps2[i]
                res.append(tmp)
        
        res = MPS(N, d, get_linkdims(res), tensors = res)
        
        if normalize:
            res = res.normalize()

        if compress:
            return res.compress(tol=tol, maxD=maxD)
        else:
            return res
           
def add_mpo(mpo1, mpo2):
    
    N = mpo1.N
    d = mpo1[0].shape[2]
    D1, D2 = get_linkdims(mpo1.tensors), get_linkdims(mpo2.tensors)

    res = []
    for i in range(N):
        if i == 0:
            a, b = D1[i + 1], D2[i + 1]
            tmp = torch.zeros(1, a + b, d, d)
            tmp[:, :a, :, :] = mpo1[i]
            tmp[:, a:, :, :] = mpo2[i]
            res.append(tmp)
        elif i == N - 1:
            a, b = D1[i], D2[i]
            tmp = torch.zeros(a + b, 1, d, d)
            tmp[:a, :, :, :] = mpo1[i]
            tmp[a:, :, :, :] = mpo2[i]
            res.append(tmp)
        else:
            a_left, b_left = D1[i], D2[i]
            a_right, b_right = D1[i + 1], D2[i + 1]
            tmp = torch.zeros(a_left + b_left, a_right + b_right, d, d)
            tmp[:a_left, :a_right, :, :] = mpo1[i]
            tmp[a_left:, a_right:, :, :] = mpo2[i]
            res.append(tmp)

    return MPO(N = N, d = 2, tensors = res)           

def get_rzz_mpo(site_1, site_2, N, theta):
        
    """
    Construct an MPO that applies a controlled Rz(theta) operation between site_1 and site_2
    where site_1 acts as the control (in Z basis), and Rz is applied at site_2.

    Parameters:
        site_1: int, first qubit (control) index
        site_2: int, second qubit (target) index
        N: total number of sites
        theta: rotation angle

    Returns:
        MPO object with N tensors
    """
    
    d = 2  # qubit system
    I = torch.eye(d, dtype=torch.cdouble)
    proj0 = torch.diag(torch.tensor([1, 0], dtype=torch.cdouble))
    proj1 = torch.diag(torch.tensor([0, 1], dtype=torch.cdouble))
    Rz = torch.diag(torch.tensor([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)], dtype=torch.cdouble))
    Rz_dag = torch.diag(torch.tensor([np.exp(1j * theta / 2), np.exp(-1j * theta / 2)], dtype=torch.cdouble))

    tensors = []
    for i in range(N):
        if i < site_1 or i > site_2 - 1:
            # Outside interaction region: identity MPO
            W = I.reshape(1, 1, d, d).clone()
        elif i == site_1:
            # Control site: projects to |0⟩⟨0| and |1⟩⟨1|
            W = torch.zeros((1, 2, d, d), dtype=torch.cdouble)
            W[0, 0] = proj0
            W[0, 1] = proj1
        elif i == site_2:
            # Target site: conditional Rz rotations depending on control
            W = torch.zeros((2, 1, d, d), dtype=torch.cdouble)
            W[0, 0] = Rz
            W[1, 0] = Rz_dag
        else:
            # Inside the interaction region: Identity on both control paths
            W = torch.zeros((2, 2, d, d), dtype=torch.cdouble)
            W[0, 0] = I
            W[1, 1] = I
        tensors.append(W)

    return MPO(N, d, tensors = tensors)
    
def get_two_site_mpo(N: int, site1: int, site2: int, op) -> List[torch.Tensor]:
    
    """
    Create MPO for Z⊗Z operator on sites `site1` and `site2`.
    
    Args:
        N (int): total number of sites
        site1 (int): first site to apply Z (0-based)
        site2 (int): second site to apply Z (0-based)

    Returns:
        List of torch.Tensor representing the MPO.
    """
        
    d = op.shape[0]
    I = torch.eye(d, dtype=torch.float64).reshape(1, 1, d, d)
    
    mpo = []

    for i in range(N):
        
        if i == site1 or i == site2:
            
            mpo.append(op.reshape(1, 1, d, d))
            
        else:
            
            mpo.append(I)
        
    return MPO(N, d, tensors = mpo)

def get_single_site_mpo(N: int, site1: int, op) -> List[torch.Tensor]:
    
    """
    Create MPO for Z⊗Z operator on sites `site1` and `site2`.
    
    Args:
        N (int): total number of sites
        site1 (int): first site to apply Z (0-based)
        site2 (int): second site to apply Z (0-based)

    Returns:
        List of torch.Tensor representing the MPO.
    """
        
    d = op.shape[0]
    I = torch.eye(d, dtype=torch.float64).reshape(1, 1, d, d)
    
    mpo = []

    for i in range(N):
        
        if i == site1:
            
            mpo.append(op.reshape(1, 1, d, d))
            
        else:
            
            mpo.append(I)
        
    return MPO(N, d, tensors = mpo)
