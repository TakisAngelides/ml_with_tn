import TensorCrossInterpolation as TCI

let

function f(x)

    # return cos(sum(x)) # simple rank case
    # return sin(5 * sum(x)) + 0.5 * cos(3 * sum(x.^2)) * exp(-0.1 * sum(x.^2)) + atan(sum(x))/5 + 0.1 * sum(x) * cos(sum(x)) # intermediate rank case
    return rand() # full rank case

end

N = 6
d = 2

localdims = fill(d, N)
tolerance = 1e-14
tci, ranks, errors = TCI.crossinterpolate2(Float64, f, localdims; tolerance = tolerance)

Isets = tci.Iset
Jsets = tci.Jset

println("Dimensions of indices")
for (element_idx, element) in enumerate(tci.sitetensors)
    println("Tensor $(element_idx): ", size(element))
end

println("-----")
for i in 1:length(Isets)-1
    println("I_$i: ", Isets[i+1])
    println("J_$(i+1): ", Jsets[i])
    println("-----")
end

end
