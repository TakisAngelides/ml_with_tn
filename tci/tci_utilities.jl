function contraction(A, c_A::Tuple, B, c_B::Tuple)::Array{ComplexF64}

    """
    The contraction function takes 2 tensors A, B and 2 tuples c_A, c_B and returns
    another tensor after contracting A and B

    A: first tensor
    c_A: indices of A to contract (Tuple of Int64)
    B: second tensor
    c_B: indices of B to contract (Tuple of Int64)

    Note 1: c_A and c_B should be the same length and the first index from c_A should
    have the same dimension as the first index of c_B, the second index from c_A
    should have the same dimension as the second index of c_B and so on.

    Note 2: It is assumed that the first index in c_A is to be contracted with the
    first index in c_B and so on.

    Note 3: If we were instead to use vectors for c_A and c_B, the memory allocation 
    sky rockets and the run time is 10 times slower. Vectors require more memory than
    tuples and run time since tuples are immutable and only store a certain type each time etc.

    Example: If A is a 4-tensor, B is a 3-tensor and I want to contract the first
    index of A with the second index of B and the fourth index of A with the first
    index of B, then the input to the contraction function should be:

    contraction(A, (1, 4), B, (2, 1))

    This will result in a 3-tensor since we have 3 open indices left after the
    contraction, namely second and third indices of A and third index of B

    Code Example:
    # @time begin
    # A = cat([1 2; 3 4], [5 6; 7 8], dims = 3)
    # B = cat([9 11; 11 12], [13 14; 15 16], dims = 3)
    # c_A = (1, 2)
    # c_B = (2, 1)
    # display(contraction(A, c_A, B, c_B))
    # end
    """

    # Get the dimensions of each index in tuple form for A and B

    A_indices_dimensions = size(A) # returns tuple(dimension of index 1 of A, ...)
    B_indices_dimensions = size(B)

    # Get the uncontracted indices of A and B named u_A and u_B. The setdiff
    # returns the elements which are in the first argument and which are not
    # in the second argument.

    u_A = setdiff(1:ndims(A), c_A)
    u_B = setdiff(1:ndims(B), c_B)

    # Check that c_A and c_B agree in length and in each of their entry they
    # have the same index dimension using the macro @assert. Below we also find
    # the dimensions of each index of the uncontracted indices as well as for the
    # contracted ones.

    dimensions_c_A = A_indices_dimensions[collect(c_A)]
    dimensions_u_A = A_indices_dimensions[collect(u_A)]
    dimensions_c_B = B_indices_dimensions[collect(c_B)]
    dimensions_u_B = B_indices_dimensions[collect(u_B)]

    @assert(dimensions_c_A == dimensions_c_B, "Note 1 in the function
    contraction docstring is not satisfied: indices of tensors to be contracted
    should have the same dimensions. Input received: indices of first tensor A
    to be contracted have dimensions $(dimensions_c_A) and indices of second
    tensor B to be contracted have dimensions $(dimensions_c_B).")

    # Permute the indices of A and B so that A has all the contracted indices
    # to the right and B has all the contracted indices to the left.

    # NOTE: The order in which we give the uncontracted indices (in this case
    # they are in increasing order) affects the result of the final tensor. The
    # final tensor will have indices starting from A's indices in increasing
    # ordera and then B's indices in increasing order. In addition c_A and c_B
    # are expected to be given in such a way so that the first index of c_A is
    # to be contracted with the first index of c_B and so on. This assumption is
    # crucial for below, since we need the aforementioned specific order for
    # c_A, c_B in order for the vectorisation below to work.

    A = permutedims(A, (u_A..., c_A...)) # Splat (...) unpacks a tuple in the argument of a function
    B = permutedims(B, (c_B..., u_B...))

    # Reshape tensors A and B so that for A the u_A are merged into 1 index and
    # the c_A are merged into a second index, making A essentially a matrix.
    # The same goes with B, so that A*B will be a vectorised implementation of
    # a contraction. Remember that c_A will form the columns of A and c_B will
    # form the rows of B and since in A*B we are looping over the columns of A
    # with the rows of B it is seen from this fact why the vectorisation works.

    # To see the index dimension of the merged u_A for example you have to think
    # how many different combinations I can have of the individual indices in
    # u_A. For example if u_A = (2, 4) this means the uncontracted indices of A
    # are its second and fourth index. Let us name them alpha and beta
    # respectively and assume that alpha ranges from 1 to 2 and beta from
    # 1 to 3. The possible combinations are 1,1 and 1,2 and 1,3 and 2,1 and 2,2
    # and 2,3 making 6 in total. In general the total dimension of u_A will be
    # the product of the dimensions of its indivual indices (in the above
    # example the individual indices are alpha and beta with dimensions 2 and
    # 3 respectively so the total dimension of the merged index for u_A will
    # be 2x3=6).

    A = reshape(A, (prod(dimensions_u_A), prod(dimensions_c_A)))
    B = reshape(B, (prod(dimensions_c_B), prod(dimensions_u_B)))

    # Perform the vectorised contraction of the indices

    C = A*B

    # Reshape the resulting tensor back to the individual indices in u_A and u_B
    # which we previously merged. This is the unmerging step.

    C = reshape(C, (dimensions_u_A..., dimensions_u_B...))

    return C

end

function lu_full_pivoting(A, max_pivots, tolerance, all_rows, all_cols)

    """
    Perform the LDU factorization of A, at the end we should have A = transpose(P)*L*D*U*transpose(Q), where 
    P and Q are the permutation matrices for the required row and column swaps during the Gaussian elimination 
    of the LU decomposition. The inverse of P and Q is their transpose; formally the decomposition yields
    P*A*Q = LU, and then we extra D from U.

    Inputs:
    A = matrix to be LDU factorized
    max_pivots = stopping condition on the factorization if max_pivots < rank of A
    tolerance = stopping condition on the factorization if the absolute value of a given pivot is below this tolerance
    all_rows, all_cols = specifies the row and column basis of A in multi-index notation

    Return: 
    P = permutation matrix for the row swaps during Gaussian elimination of the LU factorization
    L_truncated, D_truncated, U_truncated = the decompoisition matrices truncated according to max_pivots or tolerance
    Q = permutation matrix for the column swaps during Gaussian elimination of the LU factorization
    new_row_pivots = row multi-indices for the new pivots
    new_col_pivots = column multi-indices for the new pivots
    pivot_error = an approximation of the error of the LDU factorization
    """

    n, m = size(A) # row and column size of A

    L = Matrix{ComplexF64}(I, n, n) # We will mutate this to an lower triangular matrix via Gaussian elimination
    U = copy(A) # We will mutate this to an upper triangular matrix via Gaussian elimination
    P = Matrix{ComplexF64}(I, n, n) # Keeping track of the row swaps we need to do, hence this is the permutation matrix for the row swaps
    Q = Matrix{ComplexF64}(I, m, m) # Keeping track of the column swaps we need to do, hence this is the permutation matrix for the column swaps
    new_row_pivots = [] # We will store the multi-indices corresponding to the rows of the newly found pivots
    new_col_pivots = [] # We will store the multi-indices corresponding to the columns of the newly found pivots
    
    pivot_error = 0 # The estimated error of the LU factorization which we update in the while loop
    col = 1 # The column to perform Gaussian elimination for (updated in the while loop)
    pivots = 0 # Number of pivots found (updated in the while loop)
    max_rank = min(max_pivots, size(A, 1), size(A, 2)) # The Gaussian elimination should stop if we found the specified number of pivots or reached the actual maximum rank of A which is min(size(A, 1), size(A, 2))
    
    # While loop to perform Gaussian elimination for the U matrix which builds up the LU factorization of A
    while pivots < max_rank

        # Find the position of the pivot in the submatrix specified by U[col:end, col:end]
        largest_element_row, largest_element_col = Tuple(argmax(abs2.(U[col:end, col:end]))) # find position of largest element in column col     
        largest_element_row, largest_element_col = largest_element_row + col - 1, largest_element_col + col - 1

        # Stopping condition on the magnitude of the pivot compared to the input tolerance
        pivot_error = abs2(U[largest_element_row, largest_element_col]) # Estimate the error of the LU factorization as the last pivot we searched for
        if pivot_error <= tolerance # Note if this is not satisfied, we will use the pivot that we assigned to pivot_error (this is why pivot_error is an estimate, because the actual error would not be what we account for)
            break
        end
        
        # Insert the multi-index of the row of the pivot found to the new row pivots
        push!(new_row_pivots, all_rows[largest_element_row])

        # Peform a row swap on U, P and L matrices if the new pivot is not on row = col, here we also update the all_rows list which has an order of elements specifying the row basis of U
        if largest_element_row != col # case of row swap needed
            all_rows[largest_element_row], all_rows[col] = all_rows[col], all_rows[largest_element_row] 
            U[largest_element_row, :], U[col, :] = U[col, :], U[largest_element_row, :] # swaps rows largest_element_row with col in U
            P[largest_element_row, :], P[col, :] = P[col, :], P[largest_element_row, :] # swaps rows largest_element_row with col in P
            if col > 1
                L[largest_element_row, 1:col-1], L[col, 1:col-1] = L[col, 1:col-1], L[largest_element_row, 1:col-1] # swaps rows (only under diagonal) largest_element_row with col in L
            end
        end

        # Insert the multi-index of the row of the pivot found to the new row pivots
        push!(new_col_pivots, all_cols[largest_element_col])

        # Peform a row swap on the U and Q matrices if the new pivot is not on column = col, here we also update the all_cols list which has an order of elements specifying the column basis of U
        if largest_element_col != col # case of col swap needed
            all_cols[largest_element_col], all_cols[col] = all_cols[col], all_cols[largest_element_col]
            U[:, largest_element_col], U[:, col] = U[:, col], U[:, largest_element_col] # swaps cols largest_element_col with col in U
            Q[:, largest_element_col], Q[:, col] = Q[:, col], Q[:, largest_element_col] # swaps cols largest_element_col with col in Q
        end

        # Perform the Gaussian elimination step that transitions the L and U matrices towards lower and upper triangular
        multiplicative_factors = U[col+1:end, col] ./ U[col, col] # compute the vector holding the multiplicative factors for each row below row = col
        L[col+1:end, col] = multiplicative_factors # update L
        U[col+1:end, :] = U[col+1:end, :] .- multiplicative_factors * transpose(U[col, :]) # update U by subtracting from all rows the multiplicative factors time row = col

        # Increment the number of pivots found
        pivots += 1

        # In the next while loop iteration we perform Gaussian elimination on the next column
        col += 1

    end

    # Truncate the L and U matrices to the given bond dimension
    L_truncated = L[:, 1:col-1] # here it is col-1 and not col because at the end of every loop of the while loop above we increase col by 1
    U_truncated = U[1:col-1, :]

    # In the case of no truncation there is no error in the LU factorization
    if pivots == min(n, m)
        pivot_error = 0
    end

    # Extract the D matrix from the U matrix and change the U matrix accordingly such that we have LU -> LDU
    D_truncated = Diagonal(vcat(diag(U_truncated), ones(max(0, size(U_truncated, 1) - length(diag(U_truncated))))))
    Dinv = Diagonal(vcat(diag(U_truncated).^-1, ones(max(0, size(U_truncated, 1) - length(diag(U_truncated))))))
    U_truncated = Dinv * U_truncated

    return P, L_truncated, D_truncated, U_truncated, Q, new_row_pivots, new_col_pivots, pivot_error

end

function fixed_dinary_to_decimal(dinary::Vector{Int}, d::Int)
    if d < 2
        error("Base d must be at least 2.")
    end
    
    decimal = 0
    power = length(dinary) - 1
    dinary = dinary .- 1
    
    for digit in dinary
        if digit < 0 || digit >= d
            error("Digits must be in the range 0 to $(d-1).")
        end
        decimal += digit * d^power
        power -= 1
    end

    return decimal
end

function decimal_to_fixed_dinary(n::Int, d::Int, N::Int)
    digits = []
    for _ in 1:N
        push!(digits, n % d)
        n ÷= d
    end
    return reverse(digits)
end

function func_mps(dinary, min_grid, max_grid, d)

    """
    Convert a dinary vector to a grid value and then evaluate a given function with that input value

    Inputs:
    ditstring = dinary vector of integers between 1 and d e.g. [3, 2, 3, 1] for d = 3
    min_grid, max_grid = the grid domain over which values are mapped

    Return:
    x, y = grid value, function value at that grid value input
    """

    N = length(dinary)

    delta = (max_grid - min_grid)/(d^N - 1)

    x = min_grid + fixed_dinary_to_decimal(dinary, d) * delta

    # return x, (sin(5x) + 0.5cos(3x^2)) * exp(-0.1x^2) + atan(x)/5 + 0.1 * x * cos(x)
    
    # return x, sin(x)^2 + exp(-x^2) * cos(3x) + log(1 + x^2)

    # return x, tanh(sin(x) + cosh(0.5x)) + atan(x^3) - exp(-abs(x))

    # return x, log1p(sin(x)^2 + x^2) + exp(-x^2/5) * cos(5x)

    # return x, cos(x^2) * exp(-abs(x)) + sinh(x)/cosh(x + 1)

    # return x, sin(x^3) / sqrt(x^2 + 1) + log(1 + x^4)

    # return x, x

    return x, sin(x)

    # return x, cos(x)

    # return x, exp(-(x^2)/(0.1))

    # return x, sin(1/x)

    # return x, 1

    # return x, 10*sin(2π*x/365 - π/2) + 15

    # return x, sin(x)

end

function func_qft(dinary, min_grid, max_grid, d)

    """
    Convert a dinary vector to a grid value and then evaluate a given function with that input value

    Inputs:
    ditstring = dinary vector of integers between 1 and d e.g. [3, 2, 3, 1] for d = 3
    min_grid, max_grid = the grid domain over which values are mapped

    Return:
    x, y = grid value, function value at that grid value input
    """

    N = length(dinary)

    delta = (max_grid - min_grid)/(d^N - 1)

    x = min_grid + fixed_dinary_to_decimal(dinary, d) * delta

    sum_res = 0
    for l in 1:N
        for l_dash in 1:N
            sigma = decimal_to_fixed_dinary(dinary[l]-1, 2, 2)[2]
            sigma_l_dash = decimal_to_fixed_dinary(dinary[end-l_dash+1]-1, 2, 2)[1]
            sum_res += 2.0^(N-l_dash-l) * sigma * sigma_l_dash
        end
    end

    res = (1/sqrt(sqrt(d)^(N)))*exp(-1im * 2 * pi * sum_res)

    return x, res

end

function get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, min_grid, max_grid, max_pivots, tolerance, func, func_cache, d)

    """
    Constructs the matrix Pi for a given level `l`, performs LDU decomposition, and retrieves updated pivot info.

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
    Pi = zeros(ComplexF64, length(row_pivots_l), d, d, length(col_pivots_l))
    flag = true
    for row_s_idx in 1:d # The order of the for loops for the rows here is important to match the basis order stored in all_rows, all_cols with the reshape of the Pi tensor done below
        for (row_idx, row) in enumerate(row_pivots_l)
            push!(all_rows, vcat(row, row_s_idx))
            for (col_idx, col) in enumerate(col_pivots_l) # The order of the for loops for the columns here is important to match the basis order stored in all_rows, all_cols with the reshape of the Pi tensor done below
                for col_s_idx in 1:d
                    if flag # Flag is set to true only for the first round of looping over the columns, once the second row starts we set this to false as we have inserted all columns in the all_cols list
                        push!(all_cols, vcat(col_s_idx, col))
                    end
                    dinary = vcat(row, row_s_idx, col_s_idx, col)
                    if haskey(func_cache, dinary) # If the function has already been evaluated at a given dinary we get the value from a dictionary cache
                        x, y = func_cache[dinary]
                    else
                        x, y = func(dinary, min_grid, max_grid, d)
                        func_cache[dinary] = (x, y)
                    end
                    Pi[row_idx, row_s_idx, col_s_idx, col_idx] = y # Build the Pi tensor to be LDU decomposed
                end
            end
            flag = false
        end
    end
    # Reshape uses column major order so the left most index changes faster hence the ordering of the for loops above for the rows and columns 
    Pi = reshape(Pi, length(row_pivots_l)*d, d*length(col_pivots_l))
    
    P, L, D, U, Q, new_row_pivots, new_col_pivots, pivot_error = lu_full_pivoting(Pi, max_pivots, tolerance, all_rows, all_cols)

    # These matrices are needed for the assignment of tensors into the MPS in CI canonical form
    L11, L21 = UnitLowerTriangular(L[1:size(L, 2), :]), L[size(L, 2)+1:end, :] # L11 = chi x chi, L21 = n - chi x chi
    U11, U12 = UnitUpperTriangular(U[:, 1:size(U, 1)]), U[:, size(U, 1)+1:end] # U11 = chi x chi, U12 = chi x m - chi

    return P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache

end

function initialize_pivots_and_cache(func, min_grid, max_grid, d, N, num_starting_pivots)

    """
    Initializes the pivot structures and function evaluation cache using num_starting_pivots random pivots.

    Inputs:
    func = target function to evaluate tensor entries
    min_grid, max_grid = domain boundaries for evaluation
    d = local physical dimension
    N = total number of sites
    num_starting_pivots = number of initial random pivot entries to generate

    Returns:
    row_pivots, col_pivots = list of multi-index row/col pivots for each site
    func_cache = dictionary cache mapping dinary inputs to function evaluations
    """

    func_cache = Dict{Vector{Int}, Tuple{Any, ComplexF64}}()

    dinarys = []
    for _ in 1:num_starting_pivots
        random_dinary = [rand(1:d) for _ in 1:N]
        push!(dinarys, random_dinary)
    end

    row_pivots = [[dinarys[1][1:l]] for l in 0:N-1]
    col_pivots = [[dinarys[1][l+1:N]] for l in 1:N]
    x, y = func(dinarys[1], min_grid, max_grid, d)
    func_cache[dinarys[1]] = (x, y)

    for i in 2:num_starting_pivots
        dinary = dinarys[i]
        for l in 0:N-1
            if l != 0
                push!(row_pivots[l+1], dinary[1:l])
            end
            if l != N-1
                push!(col_pivots[l+1], dinary[l+2:N])
            end
        end
        if haskey(func_cache, dinary)
            continue
        else
            x, y = func(dinary, min_grid, max_grid, d)
            func_cache[dinary] = (x, y)
        end
    end

    return row_pivots, col_pivots, func_cache

end

function get_mps_from_pivots(row_pivots, col_pivots, min_grid, max_grid, d, N, func_cache, max_pivots, tolerance, func)

    """
    Constructs an MPS representation from current pivot sets and function cache.

    Inputs:
    row_pivots, col_pivots = current list of row and column multi-index pivots
    min_grid, max_grid = domain bounds for function evaluation
    d = physical bond dimension
    N = number of sites
    func_cache = dictionary of cached function evaluations
    max_pivots = maximum pivot rank for LDU truncation
    tolerance = stopping threshold for pivoting error

    Returns:
    mps_list = list of tensors forming the MPS
    row_pivots, col_pivots = updated pivot sets after decompositions
    """

    t_tensors = []
    p_tensors = []

    for i in 1:N-1

        # Build the p matrices do an LU decomposition to identify small pivots and remove them while updating the pivot lists
        rows = row_pivots[i+1]
        cols = col_pivots[i]
        p = zeros(ComplexF64, length(rows), length(cols))
        for (row_idx, row) in enumerate(rows)
            for (col_idx, col) in enumerate(cols)
                dinary = vcat(row, col)
                if haskey(func_cache, dinary) 
                    x, y = func_cache[dinary]
                else
                    x, y = func(dinary, min_grid, max_grid, d)
                    func_cache[dinary] = (x, y)
                end
                p[row_idx, col_idx] = y
            end
        end
        P, L_truncated, D_truncated, U_truncated, Q, new_row_pivots, new_col_pivots, pivot_error = lu_full_pivoting(p, max_pivots, tolerance, rows, cols)
        row_pivots[i+1] = new_row_pivots
        col_pivots[i] = new_col_pivots

        # Rebuild the p matrices after the update of the pivot lists
        rows = row_pivots[i+1]
        cols = col_pivots[i]
        p = zeros(ComplexF64, length(rows), length(cols))
        for (row_idx, row) in enumerate(rows)
            for (col_idx, col) in enumerate(cols)
                dinary = vcat(row, col)
                if haskey(func_cache, dinary) 
                    x, y = func_cache[dinary]
                else
                    x, y = func(dinary, min_grid, max_grid, d)
                    func_cache[dinary] = (x, y)
                end
                p[row_idx, col_idx] = y
            end
        end
        push!(p_tensors, inv(p))
    end

    # Build the T tensors 
    for i in 1:N
        rows = row_pivots[i]
        cols = col_pivots[i]

        if i == 1
            t = zeros(ComplexF64, d, length(cols))
            for (col_idx, col) in enumerate(cols)
                for s in 1:d
                    dinary = vcat(s, col)
                    if haskey(func_cache, dinary) 
                        x, y = func_cache[dinary]
                    else
                        x, y = func(dinary, min_grid, max_grid, d)
                        func_cache[dinary] = (x, y)
                    end
                    t[s, col_idx] = y
                end
            end
        elseif i == N
            t = zeros(ComplexF64, length(rows), d)
            for (row_idx, row) in enumerate(rows)
                for s in 1:d
                    dinary = vcat(row, s)
                    if haskey(func_cache, dinary) 
                        x, y = func_cache[dinary]
                    else
                        x, y = func(dinary, min_grid, max_grid, d)
                        func_cache[dinary] = (x, y)
                    end
                    t[row_idx, s] = y
                end
            end
        else
            t = zeros(ComplexF64, length(rows), d, length(cols))
            for (row_idx, row) in enumerate(rows)
                for (col_idx, col) in enumerate(cols)
                    for s in 1:d
                        dinary = vcat(row, s, col)
                        if haskey(func_cache, dinary) 
                            x, y = func_cache[dinary]
                        else
                            x, y = func(dinary, min_grid, max_grid, d)
                            func_cache[dinary] = (x, y)
                        end
                        t[row_idx, s, col_idx] = y
                    end
                end
            end
        end
        push!(t_tensors, t)
    end

    # Initialize the MPS list
    mps_list = []
    for i in 1:N
        if i == 1
            push!(mps_list, zeros(ComplexF64, d, 1))
        elseif i == N
            push!(mps_list, zeros(ComplexF64, 1, d))
        else
            push!(mps_list, zeros(ComplexF64, 1, d, 1))
        end
    end

    # Build the MPS tensors T_{1}, P^-1_{1}*T_{2}, ..., P^-1_{N-1}*T_{N}
    mps_list[1] = t_tensors[1]
    for i in 2:N
        mps_list[i] = contraction(p_tensors[i-1], (2,), t_tensors[i], (1,))
    end
    
    return mps_list, row_pivots, col_pivots

end

function tci(N, func, min_grid, max_grid, tolerance, max_pivots, sweeps, d, func_cache, row_pivots, col_pivots, mps_list)

    """
    Performs the Tensor Cross Interpolation (TCI) algorithm to approximate a high-dimensional tensor 
    using matrix product states (MPS).

    Inputs:
    N = number of dimensions (sites)
    func = function to sample the tensor
    min_grid, max_grid = bounds of the domain
    tolerance = stopping criterion for pivot values
    max_pivots = maximum number of pivots allowed per decomposition
    sweeps = number of left-right sweeps for refinement

    Return:
    mps_list = list of MPS tensors approximating the original tensor
    pivot_error_list = per-site pivot errors during final sweep
    func_cache = dictionary of evaluated tensor entries
    """

    # Each link will be associated with an error coming from LDU decomposing the corresponding Pi tensor
    pivot_error_list = zeros(N-1)

    # We will store the MPS tensors in this list and its bond dimensions
    
    bonddims = [size(mps_list[i], 1) for i in 2:N]
    bonddims_previous = copy(bonddims)
    final_sweep = 1
  
    for sweep in 1:sweeps
        
        for dir in (true, false)  # true = forward, false = backward
        
            if dir
                range_l = 1:N-1
            else
                range_l = N-1:-1:1
            end

            for l in range_l

                # Perform the LU decomposition on site l, l+1 and get the new tensors and pivots
                P, L, L11, L21, D, U, U11, U12, Q, new_row_pivots, new_col_pivots, pivot_error, func_cache = get_decomposition_and_updated_pivots(l, row_pivots, col_pivots, min_grid, max_grid, max_pivots, tolerance, func, func_cache, d)

                if dir
                    left_tensor = transpose(P) * vcat(Matrix(I, size(L, 2), size(L, 2)), L21 * inv(L11)) # T_l * P^-1
                    right_tensor = L11 * D * U * transpose(Q) # T_{l+1}
                else
                    left_tensor = transpose(P) * L * D * U11 # T_l
                    right_tensor = hcat(Matrix(I, size(U, 1), size(U, 1)), inv(U11) * U12) * transpose(Q) # P^-1 * T_{l+1}
                end

                # Reshape tensors into their MPS form
                if l == 1
                    left_tensor = reshape(left_tensor, d, size(left_tensor, 2)) # Open boundary conditions have no left index for the first MPS tensor
                else
                    left_tensor = reshape(left_tensor, div(size(left_tensor, 1), d), d, size(left_tensor, 2)) # Indexing is: left bond index, physical index, right bond index
                end
                if l == N-1
                    right_tensor = reshape(right_tensor, size(right_tensor, 1), d) # Open boundary conditions have no right index for the last MPS tensor
                else
                    right_tensor = reshape(right_tensor, size(right_tensor, 1), d, div(size(right_tensor, 2), d))
                end

                # Update the MPS tensors
                mps_list[l] = left_tensor
                mps_list[l + 1] = right_tensor

                # Update the pivots
                row_pivots[l + 1] = new_row_pivots
                col_pivots[l] = new_col_pivots

                # Update the error
                pivot_error_list[l] = pivot_error

                # Update bond dimensions
                bonddims[l] = size(right_tensor, 1)
                
            end
        end

        # Stopping condition if the bond dimensions remain the same after a full sweep and after a minimum of 10 sweeps
        println("Sweep $(sweep): $(bonddims)")
        if bonddims == bonddims_previous && sweep > 10
            println("TCI stopped by bond dimension convergence on sweep $(sweep) with bond dimensions $(bonddims).")
            return mps_list, pivot_error_list, func_cache, sweep, bonddims
        else
            bonddims_previous = copy(bonddims)
        end

        final_sweep = sweep

    end

    println("TCI stopped by maximum number of sweeps")
    return mps_list, pivot_error_list, func_cache, final_sweep, bonddims

end

function mps_list_to_itensors_mps(N, mps_list, sites)

    # Convert mps list to ITensors MPS
    links = [Index(size(mps_list[n], 1), "Link,i=$(n-1)") for n in 2:N]
    mps_tensors = Vector{ITensor}(undef, N)
    mps_tensors[1] = ITensor(mps_list[1], sites[1], links[1])
    for n in 2:(N-1)
        mps_tensors[n] = ITensor(mps_list[n], links[n-1], sites[n], links[n])
    end
    mps_tensors[N] = ITensor(mps_list[N], links[N-1], sites[N])
    mps = MPS(mps_tensors)

    return mps

end

function mps_list_to_itensors_mpo(N, mps_list, sites, d)

    # Convert mps list to ITensors MPS

    mpo_list = []
    for i in 1:N
        if i == 1
            push!(mpo_list, reshape(mps_list[i], div(d, 2), div(d, 2), size(mps_list[i], 2)))
        elseif i == N
            push!(mpo_list, reshape(mps_list[i], size(mps_list[i], 1), div(d, 2), div(d, 2)))
        else
            push!(mpo_list, reshape(mps_list[i], size(mps_list[i], 1), div(d, 2), div(d, 2), size(mps_list[i], 3)))
        end
    end
    links = [Index(size(mpo_list[n], 1), "Link,i=$(n-1)") for n in 2:N]
    mpo_tensors = Vector{ITensor}(undef, N)
    mpo_tensors[1] = ITensor(mpo_list[1], sites[1], sites[1]', links[1]) # the dash is on the second site index because reshape above uses major column order which means left index changes fastest when opening up a d dimensional index into a d/2, d/2 double index but in the func the conversation from decimal to binary has the convention that the right index changes fastest
    for n in 2:(N-1)
        mpo_tensors[n] = ITensor(mpo_list[n], links[n-1],  sites[n], sites[n]', links[n])
    end
    mpo_tensors[N] = ITensor(mpo_list[N], links[N-1], sites[N], sites[N]')
    mpo = MPO(mpo_tensors)

    return mpo

end

function get_hadamard_mpo(sites, idx)

    opsum = OpSum()
    opsum += [1 1; 1 -1],idx
    return MPO(opsum, sites)

end

function get_phase_mpo(sites, starting_site, sign)

    # Define links indices and initialize MPO of time evolution
    N = length(sites)
    links = [n < starting_site ? Index(1, "Link,l=$n") : Index(2, "Link,l=$n") for n in 1:N-1]
    mpo = MPO(sites)

    # First tensor of MPO
    s1, s2, l1 = prime(sites[1]), dag(sites[1]), links[1]
    mpo[1] = ITensor(ComplexF64, l1, s1, s2)
    if starting_site == 1
        # |0><0|
        mpo[1][l1 => 1, s1 => 1, s2 => 1] = 1
        # |1><1|
        mpo[1][l1 => 2, s1 => 2, s2 => 2] = 1
    else
        # Id
        mpo[1][l1 => 1, s1 => 1, s2 => 1] = 1
        mpo[1][l1 => 1, s1 => 2, s2 => 2] = 1
    end 
    
    # Set the bulk MPO 
    for i in 2:N-1
        
        s1, s2, l1, l2 = prime(sites[i]), dag(sites[i]), links[i-1], links[i]
        mpo[i] = ITensor(ComplexF64, l1, l2, s1, s2)

        if i < starting_site
            # Id
            mpo[i][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1
            mpo[i][l1 => 1, l2 => 1, s1 => 2, s2 => 2] = 1   
        elseif i == starting_site
            # |0><0|
            mpo[i][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1 
            # |1><1|
            mpo[i][l1 => 1, l2 => 2, s1 => 2, s2 => 2] = 1
        else
            # Id
            mpo[i][l1 => 1, l2 => 1, s1 => 1, s2 => 1] = 1
            mpo[i][l1 => 1, l2 => 1, s1 => 2, s2 => 2] = 1
            # Phase matrix
            mpo[i][l1 => 2, l2 => 2, s1 => 1, s2 => 1] = 1
            mpo[i][l1 => 2, l2 => 2, s1 => 2, s2 => 2] = exp(sign*1im * pi / 2^(i - starting_site))
        end

    end

    # Last tensor of MPO
    s1, s2, l1 = prime(sites[N]), dag(sites[N]), links[N-1]
    mpo[N] = ITensor(ComplexF64, l1, s1, s2)
    # Id
    mpo[N][l1 => 1, s1 => 1, s2 => 1] = 1
    mpo[N][l1 => 1, s1 => 2, s2 => 2] = 1
    # Phase matrix
    mpo[N][l1 => 2, s1 => 1, s2 => 1] = 1
    mpo[N][l1 => 2, s1 => 2, s2 => 2] = exp(sign*1im * pi / 2^(N - starting_site))
    
    return mpo

end

function get_qft_mpo(sites, qft_sign_convention, cutoff)

    N = length(sites)
    mpo = get_hadamard_mpo(sites, 1)
    for i in 1:N-1
        mpo = apply(get_phase_mpo(sites, i, qft_sign_convention), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
        mpo = apply(get_hadamard_mpo(sites, i + 1), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
    end
    return mpo

end

function get_inv_qft_mpo(sites, qft_sign_convention, cutoff)

    N = length(sites)
    mpo = get_hadamard_mpo(sites, N)
    for i in N-1:-1:1
        mpo = apply(get_phase_mpo(sites, i, -qft_sign_convention), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
        mpo = apply(get_hadamard_mpo(sites, i), mpo; cutoff = cutoff) # important that the mpo is on the right in this multiplication
    end
    return mpo

end

function reverse_mps(mps, sites)
    mps_reversed = MPS(sites)
    N=length(sites)
    for i in 1:N
        mps_reversed[i] = mps[N-i+1]
    end
    return mps_reversed
end

function decimal_to_padded_binary_list(decimal, bit_length)
    binary_list = Int[]

    while decimal > 0 || length(binary_list) < bit_length
        pushfirst!(binary_list, (decimal % 2) + 1)
        decimal = div(decimal, 2)
    end

    # Pad with leading zeros if needed
    while length(binary_list) < bit_length
        pushfirst!(binary_list, 0)
    end

    return binary_list 
end

function mps_to_list(mps; reverse_flag = false)::Vector{promote_itensor_eltype(mps)}
    
    res = []
    N = length(mps)
    tmp = contract(mps)
    for i in 1:2^(N)
        # Convert a decimal into a list of integers that represent the number in binary 
        # (instead of 0 and 1 we have 1 and 2 in this list to fit with Julia)
        if reverse_flag
            binary_list = reverse(decimal_to_padded_binary_list(i-1, N))
        else
            binary_list = decimal_to_padded_binary_list(i-1, N) 
        end
        push!(res, tmp[binary_list...])
    end
    return res

end

function decimal_to_padded_dinary_list(n, d, N)

    """

    Convert a non-negative decimal integer `n` into its fixed-length base-`d` representation (a "dinary").

    The result is a vector of length `N`, representing the base-`d` digits of `n`, with the most significant digit first.
    If `n` cannot be represented using `N` digits in base `d`, an error is thrown.

    # Arguments
    - `n::Int`: A non-negative integer to convert.
    - `d::Int`: The base (must be at least 2).
    - `N::Int`: The number of digits to output (must be large enough to represent `n` in base `d`).

    # Returns
    - `Vector{Int}`: A vector of `N` integers, each in the range `0` to `d-1`, representing `n` in base `d`.

    # Example
    ```julia
    decimal_to_fixed_dinary(13, 3, 5) # returns [0, 0, 1, 1, 1]

    """
    
    result = fill(0, N)
    i = N

    while n > 0 && i > 0
        result[i] = n % d
        n ÷= d
        i -= 1
    end

    if n > 0
        error("Number too large to fit in $N digits for base $d.")
    end

    return result
end