from mps_utilities import *
from TorNcon import ncon

# Example MPS
n = 5
d = 2
linkdims = [1] + [1 for _ in range(n-1)] + [1]
mps = MPS(n, d, linkdims)

# Example MPO for Ising model
# mpo = get_ising_mpo(n = n, J = 1.0, gx = 0.5, gz = 0.1)

# dmrg = DMRG(mpo, mps, etol = 1e-8, tol = 1e-8, maxD = 40, verbose = False)
# E, gs = dmrg.run()
    
# print(E, gs) 

index_array = []
for i in range(mps.n):
    if i == 0:
        index_array.append([-i-1, 1, -i-2])
    elif i == mps.n-1:
        if i % 2 == 0:
            index_array.append([i,-i-3,-i-2])
        else:
            index_array.append([i-1,-i-3,-i-2])
    elif i % 2 == 0:
        index_array.append([i,i+1,-i-2])
    else:
        index_array.append([i,i+1,-i-2])
            
mps_con = ncon(mps.tensors, index_array).reshape([mps.d]*mps.n)

print(mps_con.shape)

