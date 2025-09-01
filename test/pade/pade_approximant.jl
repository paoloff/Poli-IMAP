N = 6;
M = 6;
maxOrder = 18;

realF = realify(sys.f, [1,2], invMultiply=true);

idx = 1;

coeffs = realF.tensors[idx, :, :];

########################################################################
########################################################################

# system of eqs for the denominator

allExponents = []

for power in 0:(M)
    allExponents = vcat(allExponents, (homog_exponents(2, power)));
end

numberOfUnkns = length(allExponents);

admissiblePairs = [];

for power in (N + 1):(N + M)
    admissiblePairs = vcat(admissiblePairs, reverse(homog_exponents(2, power)));
end

numberOfEqs = length(admissiblePairs);

C = zeros(ComplexF64, numberOfEqs, numberOfUnkns);

for (eqIdx, pair) in enumerate(admissiblePairs)

    α, β = pair .- 1;

    for k in 0:M, l in 0:k

        if β - l < 0 ||  α - k + l < 0
            continue
        else
            colIndex = findfirst(==((l + 1, k - l + 1)), allExponents);
            C[eqIdx, colIndex] += coeffs[α - k + l + 1, β - l + 1];
        end

    end

end

########################################################################
########################################################################

# find denominator b

s = svd(C);

zeroIdx = findfirst(<(1e-4), s.S); #argmin(s.S); 

b = s.Vt'[:, zeroIdx];

b[abs.(b).<1e-12] .= 0.0;

########################################################################
########################################################################

# find numerator a

admissiblePairs_2 = [];

for power in 0:N
    admissiblePairs_2 = vcat(admissiblePairs_2, reverse(homog_exponents(2, power)));
end

a = zeros(ComplexF64, N + 1, N + 1);

for pair in (admissiblePairs_2)

    α, β = pair .- 1;

    for k in 0:M, l in 0:k

        if β - l < 0 ||  α - k + l < 0
            continue
        else
            colIndex = findfirst(==((l + 1, k - l + 1)), allExponents);
            a[α + 1, β + 1] += coeffs[α - k + l + 1, β - l + 1] * b[colIndex];
            
        end
    end
end

########################################################################
########################################################################

# compare a/b with original series

function denominator(x::Vector{ComplexF64})

    return sum([b[i] * x[1] ^ (allExponents[i][1] - 1) *  x[2] ^ (allExponents[i][2] - 1)
    for i in eachindex(b)])

end

function numerator(x::Vector{ComplexF64})

    return sum([a[tuple[1], tuple[2]] * x[1] ^ (tuple[1] - 1) *  x[2] ^ (tuple[2] - 1)
    for tuple in Tuple(CartesianIndices(a))])

end

# x = rand(ComplexF64, sys.f.domainDim)
x = rand(Float64, sys.f.domainDim) .+ 0.0im
        
evaluate(coeffs, x)

numerator(x)/denominator(x)
