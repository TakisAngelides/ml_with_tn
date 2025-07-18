from mps_utilities import *
import matplotlib.pyplot as plt
from ttn_utilities import *

# ------------------------------------------------------------------------------------------------------------------------------

# # MPS
# # Both area law and volume states will have a decaying connected 2 point function - https://quantumghent.github.io/TensorTutorials/3-MatrixProductStates/InfiniteMPS.html#imps-correlation

# N = 100
# d = 2
# D = 100

# # Z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))
# # mpo = get_two_site_mpo(N, 0, 2, Z)
# # print(mpo.to_matrix())

# linkdims = [1] + [D]*(N-1) + [1]
# mps = MPS(N, d, linkdims = linkdims).compress().normalize()

# # t = mps[98]
# # tconj = t.conj()
# # s1, s2 = t.shape, tconj.shape
# # res = torch.tensordot(t, tconj, [[2], [2]]).permute(0, 2, 1, 3).reshape(s1[0]*s2[0], s1[1]*s2[1])

# # evals, evecs = torch.linalg.eig(res)
# # print(evals)

# dmrg = DMRG(get_ising_mpo(N, J = -1, gx = 0.2, gz = 0.1), mps, verbose = True, tol = 1e-9, etol = 1e-9)
# E0, mps = dmrg.run()

# Z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))

# corr = []
# dist = range(1, 10)
# site1 = 20

# for i in dist:
    
#     site2 = site1 + i
    
#     mpo1 = get_two_site_mpo(N, site1, site2, Z)
#     mpo2 = get_single_site_mpo(N, site1, Z)
#     mpo3 = get_single_site_mpo(N, site2, Z)
#     corr.append(abs(mps.get_expectation_value(mpo1).item() - mps.get_expectation_value(mpo2).item()*mps.get_expectation_value(mpo3).item()))
    
# plt.plot(dist, corr)
# plt.show()

# ------------------------------------------------------------------------------------------------------------------------------

# TTN

depth = 8
N = 2**depth
d = 2
D = 10

ttn = random_ttn(N, d, D).canonicalize(normalize = True)

sites = ttn.get_physical_indices()

Z = torch.diag(torch.tensor([1.0, -1.0], dtype=torch.float64))

corr = []
dist = range(1, 10)
site1 = 10

for i in dist:
    
    site2 = site1 + i
    
    mpo1 = get_two_site_mpo(N, site1, site2, Z, physical_indices = sites)
    mpo2 = get_single_site_mpo(N, site1, Z, physical_indices = sites)
    mpo3 = get_single_site_mpo(N, site2, Z, physical_indices = sites)
    
    corr.append(abs(ttn.expectation_value(mpo1) - ttn.expectation_value(mpo2)*ttn.expectation_value(mpo3)))
    
plt.plot(dist, corr)
plt.show()

# ------------------------------------------------------------------------------------------------------------------------------
