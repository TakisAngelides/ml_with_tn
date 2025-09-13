import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import random as rdm
from mps_utilities import *
from PIL import Image
rdm.seed(1)

def fixed_dinary_to_decimal(dinary, d):
    
    if d < 2:
        raise ValueError("Base d must be at least 2.")
    
    decimal = 0
    power = len(dinary) - 1
    # Convert 1-based digits to 0-based

    for digit in dinary:
        if digit < 0 or digit >= d:
            raise ValueError(f"Digits must be in the range 0 to {d-1}.")
        decimal += digit * (d ** power)
        power -= 1

    return decimal

def decimal_to_fixed_dinary(n: int, d: int, N: int):
    
    if d < 2:
        raise ValueError("Base d must be at least 2.")
    if n < 0:
        raise ValueError("Decimal number must be non-negative.")
    
    result = [0] * N
    i = N - 1  # Python uses 0-based indexing
    
    while n > 0 and i >= 0:
        result[i] = n % d
        n //= d
        i -= 1
    
    if n > 0:
        raise ValueError(f"Number too large to fit in {N} digits for base {d}.")
    
    return result

def func(dinary, func_inputs):
    
    """
    Convert a dinary vector to a multi-dimensional grid coordinate vector
    and evaluate a given function at that input coordinate.

    Inputs:
        dinary: list of integers between 1 and d, length = dim * chunk_length
        d: base for the dinary vector
    
    Returns:
        x_vec: tuple of 2 integers specifying the position of the pixel
        y: function value at x_vec
    """
    
    d, img = func_inputs["d"], func_inputs["img"]

    chunk_length = len(dinary) // 2
    x_vec = []

    for i in range(2):
        # chunk = dinary[i*chunk_length:(i+1)*chunk_length]
        chunk = dinary[i::2]
        decimal_value = fixed_dinary_to_decimal(chunk, d)
        x_vec.append(decimal_value)
        
    y = img[x_vec[0], x_vec[1]]
        
    return x_vec, y

def lu_full_pivoting(A, max_pivots, tolerance, all_rows, all_cols):
    
    """
    Perform LDU factorization of A with full pivoting:
    A = P^T * L * D * U * Q^T,
    where P, Q are permutation matrices for rows and columns.

    Inputs:
        A: complex matrix (n x m)
        max_pivots: max number of pivots (stopping criterion)
        tolerance: stopping threshold on pivot magnitude
        all_rows, all_cols: lists of multi-index row and column labels

    Returns:
        P, L_truncated, D_truncated, U_truncated, Q: factorization matrices
        new_row_pivots, new_col_pivots: multi-index pivot labels
        pivot_error: estimate of factorization error
    """
    
    n, m = A.shape

    L = torch.eye(n, dtype=torch.complex128)
    U = A.clone()
    P = torch.eye(n, dtype=torch.complex128)
    Q = torch.eye(m, dtype=torch.complex128)
    new_row_pivots = []
    new_col_pivots = []

    pivots = 0
    col = 0
    max_rank = min(max_pivots, n, m)
    pivot_error = 0

    while pivots < max_rank:
        
        # Extract submatrix for pivot search
        submatrix = U[col:, col:]
        abs_sq = torch.abs(submatrix) ** 2

        # Find position of max absolute squared element in submatrix
        max_idx = torch.argmax(abs_sq)
        largest_element_row, largest_element_col = torch.unravel_index(max_idx, abs_sq.shape)
        largest_element_row += col
        largest_element_col += col

        pivot_error = abs_sq[largest_element_row - col, largest_element_col - col].item()
        if pivot_error <= tolerance:
            break

        # Append multi-index pivot info
        new_row_pivots.append(all_rows[largest_element_row])

        # Row swap if needed
        if largest_element_row != col:
            all_rows[largest_element_row], all_rows[col] = all_rows[col], all_rows[largest_element_row]

            U[[largest_element_row, col], :] = U[[col, largest_element_row], :]
            P[[largest_element_row, col], :] = P[[col, largest_element_row], :]

            if col > 0:
                L[[largest_element_row, col], :col] = L[[col, largest_element_row], :col]

        # Append column pivot info
        new_col_pivots.append(all_cols[largest_element_col])

        # Column swap if needed
        if largest_element_col != col:
            all_cols[largest_element_col], all_cols[col] = all_cols[col], all_cols[largest_element_col]

            U[:, [largest_element_col, col]] = U[:, [col, largest_element_col]]
            Q[:, [largest_element_col, col]] = Q[:, [col, largest_element_col]]

        # Gaussian elimination step
        pivot_val = U[col, col]
        multiplicative_factors = U[col + 1:, col] / pivot_val
        L[col + 1:, col] = multiplicative_factors

        U[col + 1:, :] -= multiplicative_factors.unsqueeze(1) * U[col, :].unsqueeze(0)

        pivots += 1
        col += 1

    # Truncate matrices according to number of pivots found
    L_truncated = L[:, :col]
    U_truncated = U[:col, :]

    if pivots == min(n, m):
        pivot_error = 0

    # Extract D from U_truncated diagonal
    diag_U = torch.diagonal(U_truncated)
    pad_length = max(0, U_truncated.shape[0] - diag_U.shape[0])
    D_values = torch.cat([diag_U, torch.ones(pad_length, dtype=torch.complex128)])
    D_truncated = torch.diag(D_values)

    Dinv_values = torch.cat([1.0 / diag_U, torch.ones(pad_length, dtype=torch.complex128)])
    Dinv = torch.diag(Dinv_values)

    U_truncated = Dinv @ U_truncated

    return P, L_truncated, D_truncated, U_truncated, Q, new_row_pivots, new_col_pivots, pivot_error

def get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, max_pivots, tolerance, func, func_cache, d, func_inputs):

    """
    Constructs the matrix Pi for a given level l, performs LDU decomposition, and retrieves updated pivot info.

    Inputs:
    l = level in the tensor contraction algorithm
    row_pivots, col_pivots = current list of multi-index row/col pivots
    min_grid, max_grid = domain boundaries for function evaluation
    max_pivots = truncation parameter
    tolerance = stopping tolerance for pivot selection
    func = function used to evaluate the matrix entries
    func_cache = dictionary to cache results of func

    Return:
    P, L, L11, L21, D, U, U11, U12, Q = decomposed components
    new_row_pivots, new_col_pivots = updated row and column pivots
    pivot_error = estimated decomposition error
    func_cache = updated cache dictionary
    """
    
    # We select the row and column pivots corresponding to optimizing sites l and l+1
    row_pivots_l = row_pivots[l]
    col_pivots_l = col_pivots[l+1]

    # We build the Pi tensor to be LDU decomposed, the important thing to ensure is that the all_rows, all_cols lists have the same ordering of elements as the basis of Pi when reshaped into a matrix
    all_rows = []
    all_cols = []
    
    num_rows = len(row_pivots_l)
    num_cols = len(col_pivots_l)

    # Initialize Pi tensor with complex dtype
    Pi = torch.zeros((num_rows, d, d, num_cols), dtype=torch.complex128)
    
    flag = True
    for row_idx, row in enumerate(row_pivots_l):
        for row_s_idx in range(d):  
            all_rows.append(row + [row_s_idx])
            for col_s_idx in range(d):
                for col_idx, col in enumerate(col_pivots_l):
                    if flag:
                        all_cols.append([col_s_idx] + col)
                    dinary = row + [row_s_idx, col_s_idx] + col

                    dinary_tuple = tuple(dinary)  # convert to tuple for dict key
                    if dinary_tuple in func_cache:
                        x, y = func_cache[dinary_tuple]
                    else:
                        x, y = func(dinary, func_inputs)
                        func_cache[dinary_tuple] = (x, y)

                    Pi[row_idx, row_s_idx, col_s_idx, col_idx] = y
            flag = False

    # After filling, reshape Pi : (num_rows * d, d * num_cols)
    Pi = Pi.reshape(num_rows * d, d * num_cols)
        
    P, L, D, U, Q, new_row_pivots, new_col_pivots, pivot_error = lu_full_pivoting(Pi, max_pivots, tolerance, all_rows, all_cols)

    # These matrices are needed for the assignment of tensors into the MPS in CI canonical form
    chi = L.shape[1]  
    # L11: top-left chi x chi block
    L11 = L[:chi, :]          # size: (chi, ?)
    # L21: bottom-left (n - chi) x chi block
    L21 = L[chi:, :]          # rows from chi to end, all columns

    chi_U = U.shape[0]  
    # U11: left chi x chi block
    U11 = U[:, :chi_U]         # all rows, first chi_U columns
    # U12: right chi x (m - chi) block
    U12 = U[:, chi_U:]         # all rows, columns from chi_U to end

    return P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache

def initialize_pivots_and_cache(func, d, N, num_starting_pivots, func_inputs):
    """
    Initializes the pivot structures and function evaluation cache using num_starting_pivots random pivots.

    Inputs:
    func = target function to evaluate tensor entries
    min_grid, max_grid = domain boundaries for evaluation (lists for each dimension)
    d = local physical dimension
    N = total number of sites
    num_starting_pivots = number of initial random pivot entries to generate

    Returns:
    row_pivots, col_pivots = list of multi-index row/col pivots for each site
    func_cache = dictionary cache mapping dinary tuple inputs to function evaluations
    """

    func_cache = dict()

    # Generate random dinarys: list of lists of length N, each digit in [0, d-1]
    dinarys = [[rdm.randint(0, d-1) for _ in range(N)] for _ in range(num_starting_pivots) ]

    # Initialize pivots as lists of lists
    row_pivots = [[] for _ in range(N)]
    col_pivots = [[] for _ in range(N)]

    # Initialize with first random pivot
    for l in range(N):
        row_pivots[l].append(dinarys[0][:l])      # first l digits
        col_pivots[l].append(dinarys[0][l+1:])      # last N-l digits
    
    x, y = func(dinarys[0], func_inputs)
    func_cache[tuple(dinarys[0])] = (x, y)

    for i in range(1, num_starting_pivots):
        dinary = dinarys[i]
        for l in range(N):
            if l != 0:
                row_pivots[l].append(dinary[:l])
            if l != N-1:
                col_pivots[l].append(dinary[l+1:])
        key = tuple(dinary)
        if key not in func_cache:
            x, y = func(dinary, func_inputs)
            func_cache[key] = (x, y)

    return row_pivots, col_pivots, func_cache

def get_mps_from_pivots(row_pivots, col_pivots, d, N, func_cache, max_pivots, tolerance, func, func_inputs):
    """
    Constructs an MPS representation from current pivot sets and function cache using PyTorch.

    Inputs:
    row_pivots, col_pivots: current lists of multi-index row/col pivots
    min_grid, max_grid: domain bounds for function evaluation
    d: physical bond dimension
    N: number of sites
    func_cache: dictionary of cached function evaluations
    max_pivots: maximum pivot rank for LDU truncation
    tolerance: stopping threshold for pivoting error
    func: function used to evaluate tensor entries

    Returns:
    mps_list: list of tensors forming the MPS
    row_pivots, col_pivots: updated pivot sets after decompositions
    """

    p_tensors = []
    t_tensors = []

    # Loop over sites 1 to N-1
    for i in range(N - 1):
        rows = row_pivots[i + 1]
        cols = col_pivots[i]
        p = torch.zeros((len(rows), len(cols)), dtype=torch.complex128)

        # Fill matrix p with function evaluations
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                dinary = tuple(row + col)
                if dinary in func_cache:
                    _, y = func_cache[dinary]
                else:
                    _, y = func(dinary, func_inputs)
                    func_cache[dinary] = (_, y)
                p[row_idx, col_idx] = y

        # LU full pivoting decomposition (assumed provided)
        P, L_trunc, D_trunc, U_trunc, Q, new_row_pivots, new_col_pivots, pivot_error = lu_full_pivoting(
            p, max_pivots, tolerance, rows, cols
        )

        row_pivots[i + 1] = new_row_pivots
        col_pivots[i] = new_col_pivots

        # Rebuild the p matrix after pivot update
        rows = row_pivots[i + 1]
        cols = col_pivots[i]
        p = torch.zeros((len(rows), len(cols)), dtype=torch.complex128)
        for row_idx, row in enumerate(rows):
            for col_idx, col in enumerate(cols):
                dinary = tuple(row + col)
                if dinary in func_cache:
                    _, y = func_cache[dinary]
                else:
                    _, y = func(dinary, func_inputs)
                    func_cache[dinary] = (_, y)
                p[row_idx, col_idx] = y

        p_tensors.append(torch.linalg.inv(p))

    # Build the T tensors
    for i in range(N):
        rows = row_pivots[i]
        cols = col_pivots[i]

        if i == 0:
            # First site: tensor shape (d, len(cols))
            t = torch.zeros((d, len(cols)), dtype=torch.complex128)
            for col_idx, col in enumerate(cols):
                for s in range(d):
                    dinary = tuple([s] + col)
                    if dinary in func_cache:
                        _, y = func_cache[dinary]
                    else:
                        _, y = func(dinary, func_inputs)
                        func_cache[dinary] = (_, y)
                    t[s - 1, col_idx] = y

        elif i == N - 1:
            # Last site: tensor shape (len(rows), d)
            t = torch.zeros((len(rows), d), dtype=torch.complex128)
            for row_idx, row in enumerate(rows):
                for s in range(d):
                    dinary = tuple(row + [s])
                    if dinary in func_cache:
                        _, y = func_cache[dinary]
                    else:
                        _, y = func(dinary, func_inputs)
                        func_cache[dinary] = (_, y)
                    t[row_idx, s - 1] = y

        else:
            # Middle sites: tensor shape (len(rows), d, len(cols))
            t = torch.zeros((len(rows), d, len(cols)), dtype=torch.complex128)
            for row_idx, row in enumerate(rows):
                for col_idx, col in enumerate(cols):
                    for s in range(d):
                        dinary = tuple(row + [s] + col)
                        if dinary in func_cache:
                            _, y = func_cache[dinary]
                        else:
                            _, y = func(dinary, func_inputs)
                            func_cache[dinary] = (_, y)
                        t[row_idx, s - 1, col_idx] = y

        t_tensors.append(t)

    # Initialize MPS list
    mps_list = []
    for i in range(N):
        if i == 0:
            mps_list.append(torch.zeros((d, 1), dtype=torch.complex128))
        elif i == N - 1:
            mps_list.append(torch.zeros((1, d), dtype=torch.complex128))
        else:
            mps_list.append(torch.zeros((1, d, 1), dtype=torch.complex128))

    # Build MPS tensors T_1, P_1^-1 * T_2, ..., P_{N-1}^-1 * T_N
    mps_list[0] = t_tensors[0]
    for i in range(1, N):
        mps_list[i] = torch.tensordot(p_tensors[i - 1], t_tensors[i], dims=([1], [0]))

    return mps_list, row_pivots, col_pivots

def mps_list_to_custom_mps(d, N, mps_list):
            
    tensors = []
    for i, tensor in enumerate(mps_list.copy()):
        if i == 0:
            tensors.append(tensor.reshape(1, *tensor.shape).permute(0, 2, 1))
        elif i == N-1:
            tensors.append(tensor.reshape(*tensor.shape, 1).permute(0, 2, 1))
        else:
            tensors.append(tensor.permute(0, 2, 1))
    
    return MPS(N, d, tensors = tensors)

def tci(N, func, tolerance, max_pivots, sweeps, d, func_cache, row_pivots, col_pivots, mps_list, func_inputs):

    # Each link will be associated with an error coming from LDU decomposing the corresponding Pi tensor
    pivot_error_list = torch.zeros(N-1)

    # We will store the MPS tensors in this list and its bond dimensions
    bonddims = [mps_list[i].shape[0] for i in range(N-1)]
    bonddims_previous = bonddims.copy()
  
    for sweep in range(sweeps):
        
        for dir in (True, False):  # true = forward, false = backward
        
            if dir:
                range_l = range(0, N-2)
            else:
                range_l = range(N-2, -1, -1)

            for l in range_l:

                # Perform the LU decomposition on site l, l+1 and get the new tensors and pivots
                P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache = get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, max_pivots, tolerance, func, func_cache, d, func_inputs)

                # Efficient and stable version
                if dir:
                    # Identity matrix matching L dimensions
                    eye_L = torch.eye(L.shape[1], dtype=L.dtype, device=L.device)

                    # Instead of L21 @ inv(L11), we solve L11 X = L21^T for X^T
                    L11_inv_L21 = torch.linalg.solve_triangular(L11.T, L21.T, upper = False, unitriangular = True).T
                    # L11_inv_L21 = L21 @ torch.linalg.inv(L11)

                    # Build left tensor
                    left_tensor = P.T @ torch.cat([eye_L, L11_inv_L21], dim=0)

                    # Build right tensor
                    right_tensor = L11 @ D @ U @ Q.T

                else:
                    # Build left tensor directly
                    left_tensor = P.T @ L @ D @ U11

                    # Identity matrix matching U dimensions
                    eye_U = torch.eye(U.shape[0], dtype=U.dtype, device=U.device)

                    # Instead of inv(U11) @ U12, we solve U11 X = U12 for X
                    U11_inv_U12 = torch.linalg.solve_triangular(U11, U12, upper = True, unitriangular = True)
                    # U11_inv_U12 = torch.linalg.inv(U11) @ U12

                    # Build right tensor
                    right_tensor = torch.cat([eye_U, U11_inv_U12], dim=1) @ Q.T

                # Reshape tensors into their MPS form
                if l == 0:
                    left_tensor = left_tensor.reshape(d, left_tensor.shape[1])
                else:
                    left_tensor = left_tensor.reshape(-1, d, left_tensor.shape[1])

                if l == N-2:
                    right_tensor = right_tensor.reshape(right_tensor.shape[0], d)
                else:
                    right_tensor = right_tensor.reshape(right_tensor.shape[0], d, -1)

                # Update the MPS tensors
                mps_list[l] = left_tensor
                mps_list[l + 1] = right_tensor

                # Update the pivots
                row_pivots[l + 1] = new_row_pivots
                col_pivots[l] = new_col_pivots

                # Update the error
                pivot_error_list[l] = pivot_error

                # Update bond dimensions
                bonddims[l] = right_tensor.shape[0]
    
        # Stopping condition if the bond dimensions remain the same after a full sweep
        if bonddims == bonddims_previous and sweep > 10:
            print(f"TCI stopped by bond dimension convergence on sweep {sweep} with bond dimensions {bonddims}.")
            return mps_list, pivot_error_list, func_cache, sweep, bonddims
        else:
            bonddims_previous = bonddims.copy()

    print("TCI stopped by maximum number of sweeps.")
    return mps_list, pivot_error_list, func_cache, sweep, bonddims


# Define parameters
N = 20
d = 2
tolerance = 1e-9
sweeps = 20
num_starting_pivots = 1

# Load image
path = "/Users/takisangelides/Downloads/img.jpg"
img = np.array(Image.open(path))

img_gray = Image.fromarray(img).convert('L')
img_resized = img_gray.resize((1024, 1024), Image.BILINEAR)
img = np.array(img_resized)
# img = img/np.max(img)

func_inputs = {"d": d, "img": img}
        
# List of max_pivots values to experiment with
max_pivots_list = [5, 10, 15, 20, 25, 30, 35, 100]

# Prepare plot
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i, max_pivots in enumerate(max_pivots_list):
    # Re-initialize pivots and cache for each run
    row_pivots, col_pivots, func_cache = initialize_pivots_and_cache(
        func, d, N, num_starting_pivots, func_inputs
    )

    mps_list, row_pivots, col_pivots = get_mps_from_pivots(
        row_pivots, col_pivots, d, N, func_cache, max_pivots, tolerance, func, func_inputs
    )

    mps_list, pivot_error_list, func_cache, sweep, bonddims = tci(
        N, func, tolerance, max_pivots, sweeps, d, func_cache,
        row_pivots, col_pivots, mps_list, func_inputs
    )

    mps_vec = torch.real(mps_list_to_custom_mps(d, N, mps_list).contract_bonds())
    order = list(range(len(mps_vec.shape)))
    mps_vec = mps_vec.permute(tuple(order[::2] + order[1::2]))
    mps_vec = mps_vec.reshape((np.prod(mps_vec.shape[0:N//2+1]), np.prod(mps_vec.shape[N//2+1:])))

    # Plot
    ax = axes[i]
    im = ax.imshow(mps_vec.T.numpy(), cmap='gray')
    ax.set_title(f"max_pivots = {max_pivots}")
    ax.axis('off')

# Adjust layout and add colorbar to last plot
plt.tight_layout()
# fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.6)
plt.show()
