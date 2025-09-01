
### Polynomial objects

mutable struct Polynomial
    # object to hold the coefficients of a polynomial of a certain order (max degree)
    # coefficients are stored in tensor format 
    order::Int64
    domainDim::Int64
    tensor::SparseArray{Complex{Float64}}
end

# empty constructor
Polynomial() = Polynomial(0, 1, SparseArray([0.0im]))

# constructor of standard polynomial
Polynomial(order::Int64, domainDim::Int64) = Polynomial(order, 
                                                        domainDim, 
                                                        SparseArray(zeros(Complex{Float64}, 
                                                        [order + 1 for i in 1:domainDim]...)))

mutable struct PolynomialArray
    # an array of Polynomials, used to represent vector or matrix-valued polynomials
    order::Int64
    domainDim::Int64
    rangeDim::Int64
    tensors::SparseArray{Complex{Float64}}
end

# empty constructor
PolynomialArray() = PolynomialArray(0, 1, 1, SparseArray([0.0im]))

# constructor of standard array
PolynomialArray(order::Int64, 
                domainDim::Int64,
                rangeDim::Int64) = PolynomialArray(order, 
                                                domainDim,
                                                rangeDim,
                                                SparseArray(zeros(Complex{Float64}, 
                                                rangeDim,
                                                [order + 1 for i in 1:domainDim]...))) 


#######################################################################################################
#######################################################################################################
#######################################################################################################

### Tape objects

mutable struct PolynomialNodeData
    pol::Union{Polynomial, Vector{Polynomial}}
    nOutputs::Int64
    constValues::Complex{Float64} # used for holding constant values
    outputIdx::Int64 # used to inform which polynomial is the output to the next node 
end

# base polynomial node data
PolynomialNodeData() = PolynomialNodeData([Polynomial()], 1, [0.], 1)

mutable struct PolynomialTapeData
    x::PolynomialArray     # input value to graphed function
    y::Vector{Polynomial}  # output value of graphed function
    iX::Int                # next input component to be processed
    iY::Int                # next output component to be processed
end

# base polynomial node data
PolynomialTapeData() = PolynomialTapeData(PolynomialArray(), [Polynomial()], 1, 1)


#######################################################################################################
#######################################################################################################
#######################################################################################################

### Polynomial functions

# number of monomials in n variables with degree k
function n_monomials(n::Int64, k::Int64)
    return div(factorial(n - 1 + k), factorial(n - 1) * factorial(k))
end


# combinations of indices up to an order with a given constant sum
function homog_exponents(n::Int64, k::Int64)
    indices = 0:k
    allCombinations = product(fill(indices, n)...)
    return reverse([reverse(comb) .+ 1 for comb in allCombinations if sum(comb) == k])
end

function homog_exponents(N::Vector{Int64}, K::Vector{Int64})

    if length(N) != length(K)
        error("Both lists must have the same length")
    end

    allCombinations = []

    for i in eachindex(N)
        indices = 0:K[i]
        currentCombinations = product(fill(indices, N[i])...)
        push!(allCombinations, reverse([reverse(comb) .+ 1 
        for comb in currentCombinations if sum(comb) == K[i]]))
    end

    allTuples = collect(product([allCombinations[i] for i in 1:length(N)]...))
    allTuplesList = []

    for tuple in allTuples
        push!(allTuplesList, Tuple(Iterators.flatten(tuple)))
    end

    return allTuplesList

end


# returns the homogenous polynomial of order k of a Polynomial
function homog_components(P::Polynomial, homogExponents::Vector)
    
    homogComponents =  SparseArray(zeros(ComplexF64, length(homogExponents)))

    for i in eachindex(homogComponents)
        homogComponents[i] = P.tensor[homogExponents[i]...]
    end

    return homogComponents
end

function homog_components(P::Vector{Polynomial}, homogExponents::Vector)

    homogComponents =  SparseArray(zeros(ComplexF64, length(P), length(homogExponents)))

    for i in eachindex(homogExponents), axis in eachindex(P)
        homogComponents[axis, i] = P[axis].tensor[homogExponents[i]...]
    end

    return homogComponents
end

function homog_components(P::PolynomialArray, homogExponents::Vector)

    homogComponents =  SparseArray(zeros(ComplexF64, P.rangeDim, length(homogExponents)))

    for i in eachindex(homogExponents), axis in 1:P.rangeDim
        homogComponents[axis, i] = P.tensors[axis, homogExponents[i]...]
    end

    return homogComponents
end


# add the coefficients inside a PolynomialArray to another PolynomialArray
function update!(P::PolynomialArray, Q::PolynomialArray, 
    homogExponents::Vector; addToCurrent::Bool=false)

    if addToCurrent
        for exponent in homogExponents, axis in 1:P.rangeDim
            P.tensors[axis, exponent...] += Q.tensors[axis, exponent...]
        end
    else
        for exponent in homogExponents, axis in 1:P.rangeDim
            P.tensors[axis, exponent...] = Q.tensors[axis, exponent...]
        end
    end

end

function update!(P::PolynomialArray, V::Vector{Polynomial}, 
    homogExponents::Vector; addToCurrent::Bool=false)

    if addToCurrent
        for exponent in homogExponents, axis in 1:P.rangeDim
            P.tensors[axis, exponent...] += V[axis].tensor[exponent...]
        end
    else
        for exponent in homogExponents, axis in 1:P.rangeDim
            P.tensors[axis, exponent...] = V[axis].tensor[exponent...]
        end
    end

end

function update!(P::PolynomialArray, M::SparseArray{Complex{Float64}}, 
    homogExponents::Vector; addToCurrent::Bool=false)

    if addToCurrent
        for (exponentIdx, exponent) in enumerate(homogExponents), axis in 1:P.rangeDim
            P.tensors[axis, exponent...] += M[axis, exponentIdx]
        end
    else
        for (exponentIdx, exponent) in enumerate(homogExponents), axis in 1:P.rangeDim
            P.tensors[axis, exponent...] = M[axis, exponentIdx]
        end
    end

end

function update!(P::PolynomialArray, M::SparseArray{Complex{Float64}}, 
    homogExponents::Vector, homogExponentsIdxs::Vector, axisIdxs::Vector{Int64}; 
    addToCurrent::Bool)

    if addToCurrent
        for (exponentIdx, exponent) in zip(homogExponentsIdxs, homogExponents), 
            axis in axisIdxs

            P.tensors[axis, exponent...] += M[axis, exponentIdx]
        end
    else
        for (exponentIdx, exponent) in zip(homogExponentsIdxs, homogExponents), 
            axis in axisIdxs

            P.tensors[axis, exponent...] = M[axis, exponentIdx]
        end
    end        

end

function update!(P::PolynomialArray, M::SparseArray{Complex{Float64}}, 
    homogExponents::Vector, axisIdxs::Vector{Int64}; 
    addToCurrent::Bool)

    if addToCurrent
        for (exponentIdx, exponent) in enumerate(homogExponents), 
            axis in axisIdxs

            P.tensors[axis, exponent...] += M[axis, exponentIdx]
        end
    else
        for (exponentIdx, exponent) in enumerate(homogExponents), 
            axis in axisIdxs

            P.tensors[axis, exponent...] = M[axis, exponentIdx]
        end
    end        

end


# function for computing the gradient a polynomial
function ∇(P::Polynomial, allHomogExponents::Vector, k::Int64)
    # this function will only compute the gradient of a Polynomial using 
    # its order k coefficients, producing a order k - 1 PolynomialArray. 
    # the reason is to save unncessary computations of gradients that
    # have already been computed in the main routine.

    # note this is a real gradient, i.e., each component of the resulting vector
    # refers to the direction where the derivative of the Polynomial function
    # is taken.

    # start tensor storing the derivatives using array constructor
    derivTensors = PolynomialArray(P.order, P.domainDim, P.domainDim)

    for derivAxes in 1:P.domainDim
        # chose axis where derivative will be taken
        discounts = zeros(Int, P.domainDim)
        discounts[derivAxes] = 1

        # add the contributions to coefficients with the same exponent
        for exponents in allHomogExponents[k + 1]
            newExponents = map(-, exponents, Tuple(discounts))
            if 0 in newExponents
                continue
            else
                derivTensors.tensors[derivAxes, newExponents...] +=
                (exponents[derivAxes] - 1) * P.tensor[exponents...]
            end
        end
    end

    return derivTensors
end

function ∇(P::Polynomial, allHomogExponents::Vector, k::Vector{Int64})

    derivTensors = PolynomialArray(P.order, P.domainDim, P.domainDim)
    
    if length(k) == 2
        for derivAxes in 1:P.domainDim
            discounts = zeros(Int, P.domainDim)
            discounts[derivAxes] = 1

            for exponents1 in allHomogExponents[1][k[1] + 1], 
                exponents2 in allHomogExponents[2][k[2] + 1]

                exponents = map(+, exponents1, exponents2 .- 1)
                newExponents = map(-, exponents, Tuple(discounts))

                if 0 in newExponents
                    continue
                else
                    derivTensors.tensors[derivAxes, newExponents...] +=
                    (exponents[derivAxes] - 1) * P.tensor[exponents...]
                end
            end
        end
    else
        for derivAxes in 1:P.domainDim

            discounts = zeros(Int, P.domainDim)
            discounts[derivAxes] = 1

            for multiExpoVectors in Iterators.product([
                allHomogExponents[i][k[i] + 1] for i in eachindex(k)]...)

                exponents = Tuple(sum([expoVector .- 1 for expoVector in multiExpoVectors]) .+ 1)
                newExponents = map(-, exponents, Tuple(discounts))

                if 0 in newExponents
                    continue
                else
                    derivTensors.tensors[derivAxes, newExponents...] +=
                    (exponents[derivAxes] - 1) * P.tensor[exponents...]
                end
            end
        end
    end

    return derivTensors
end

function ∇(P::PolynomialArray, allHomogExponents::Vector, k::Int64)

    derivTensors = PolynomialArray(P.order, P.domainDim, P.rangeDim*P.domainDim)
    axisPairs = product(1:P.rangeDim, 1:P.domainDim)

    for (axisPairIdx, axisPair) in enumerate(axisPairs)
        # unpack range and derivative axis
        rangeAxes, derivAxes = axisPair

        # differentiate (lower exponent) a tuple
        discounts = zeros(Int, P.domainDim)
        discounts[derivAxes] = 1

        # add the contributions to coefficients with the same exponent
        for exponents in allHomogExponents[k + 1]

            newExponents = map(-, exponents, Tuple(discounts))

            if 0 in newExponents
                continue
            else
                derivTensors.tensors[axisPairIdx, newExponents...] += 
                (exponents[derivAxes] - 1) * P.tensors[rangeAxes, exponents...]
            end
        end
    end

    return derivTensors
end

function ∇(P::PolynomialArray, allHomogExponents::Vector, k::Vector{Int64})

    derivTensors = PolynomialArray(P.order - 1, P.domainDim, P.rangeDim*P.domainDim)
    axisPairs = product(1:P.rangeDim, 1:P.domainDim)

    if length(k) == 2

        for (axisPairIdx, axisPair) in enumerate(axisPairs)
            # unpack range and derivative axis
            rangeAxes, derivAxes = axisPair

            # differentiate (lower exponent) a tuple
            discounts = zeros(Int, P.domainDim)
            discounts[derivAxes] = 1

            # add the contributions to coefficients with the same exponent
            for exponents1 in allHomogExponents[1][k[1] + 1], 
                exponents2 in allHomogExponents[2][k[2] + 1]

                exponents = map(+, exponents1, exponents2 .- 1)
                newExponents = map(-, exponents, Tuple(discounts))

                if 0 in newExponents
                    continue
                else
                    derivTensors.tensors[axisPairIdx, newExponents...] += 
                    (exponents[derivAxes] - 1) * P.tensors[rangeAxes, exponents...]
                end
            end
        end
    else
        for (axisPairIdx, axisPair) in enumerate(axisPairs)

            rangeAxes, derivAxes = axisPair
            discounts = zeros(Int, P.domainDim)
            discounts[derivAxes] = 1

            for multiExpoVectors in Iterators.product([
                allHomogExponents[i][k[i] + 1] for i in eachindex(k)]...)

                # exponents = Tuple(sum([expoVector .- 1 for expoVector in multiExpoVectors]) .+ 1)
                exponents = map(+, multiExpoVectors...) .- (length(k) - 1)

                newExponents = map(-, exponents, Tuple(discounts))
                
                if 0 in newExponents
                    continue
                else
                    derivTensors.tensors[axisPairIdx, newExponents...] += 
                    (exponents[derivAxes] - 1) * P.tensors[rangeAxes, exponents...]
                end
            end
        end

    end
    
    return derivTensors
end


# evaluate polynomial at point x.
# the tensor of the polynomial is given as input
function evaluate(tensor::SparseArray, x::Vector{Complex{Float64}})
    # loop over all indices of tensor using their Cartesian form.
    # accumulate products in monomial,
    # accumulate sum of monomials in y
    y = 0.0

    for expoVector in CartesianIndices(tensor)
        monomial = 1
        for i in eachindex(x)
            monomial *= x[i] ^ (expoVector[i] - 1)
        end        

        y += tensor[expoVector] * monomial        
    end

    return y
end

function evaluate(P::Polynomial, x::Vector{Complex{Float64}})
    return evaluate(P.tensor, x)
end

function evaluate(P::PolynomialArray, x::Vector{Complex{Float64}})
    return [evaluate(P.tensors[i], x) for i in 1:P.rangeDim]
end


# realify (find a re-parametrization with real coefficients) the order k components of a 
# polynomial array with complex coefficients. 
function realify(P::PolynomialArray, homogExponents::Vector, 
    cplxEigsIdxs::Vector; realEigsIdxs::Vector=[], 
    invMultiply::Bool=false, makeReal::Bool=false)
    
    # number of s components with complex eigenvalues
    Nc = length(cplxEigsIdxs)
    Nr = length(realEigsIdxs)

    # result of realification
    realP = PolynomialArray(P.order, P.domainDim, P.rangeDim)

    for axis in 1:P.rangeDim, expoVector in homogExponents

        # vector containing only the exponents corresponding to the complex
        # components
        cplxExponents = zeros(Int, Nc)

        # for each complex s, add its exponent to cplxExponents
        for (i, cplxIdx) in enumerate(cplxEigsIdxs)
            cplxExponents[i] = expoVector[cplxIdx]
        end
            
        # for each complex conjugate pair of coordinates s₁ and s₂, 
        # replace them as s₁ -> (s₁-is₂)/2 and s₂ -> (s₁+is₂)/2 .
        # then, store coefficients of binomial expansion of [(s₁-is₂)/2]ⁿ in coefficients1
        # and the coefficients of [(s₁+is₂)/2]ᵐ in coefficients2, where
        # n and m are the original exponents of s₁ and s₂. finally,  collect all the 
        # coefficients resulting from the product [(s₁-is₂)/2]ⁿ[(s₁+is₂)/2]ᵐ
        # and store them in cplxCoeffs, while also storing the combinations
        # of exponents (n', m') of the monomials s₁ⁿ's₂ᵐ' in cplxCombs

        cplxCombs = []
        cplxCoeffs = []

        for i in 1:2:(Nc - 1)
            
            coefficients1 = zeros(ComplexF64, cplxExponents[i])

            for j in 1:cplxExponents[i]
                coefficients1[j] = (1/2) ^ (cplxExponents[i] - 1) *
                ((-1im) ^ (j - 1)) * binomial((cplxExponents[i] - 1), j - 1)
            end

            coefficients2 = zeros(ComplexF64, cplxExponents[i + 1])

            for j in 1:cplxExponents[i + 1]
                coefficients2[j] = (1/2) ^ (cplxExponents[i + 1] - 1) *
                ((1im) ^ (j - 1)) * binomial((cplxExponents[i + 1] - 1), j - 1)
            end
            
            coefficientsProduct = coefficients1 * transpose(coefficients2)
            coefficients = left_diagonal_sums(coefficientsProduct)
            push!(cplxCombs, homog_exponents(2, cplxExponents[i] + cplxExponents[i + 1] - 2))
            push!(cplxCoeffs, coefficients)
        end
        # now take product between all expanded two-product terms (s₁-is₂)ⁿ(s₁+is₂)ᵐ            
        prodCoeffs = collect(product(cplxCoeffs...))
        prodExponents = collect(product(cplxCombs...))

        for (prodCoeff, prodExpoVector) in zip(prodCoeffs, prodExponents)
            newExpoVector = collect(expoVector)
            flatProdExpoVector = Tuple(Iterators.flatten(prodExpoVector))

            for (i, cplxIdx) in enumerate(cplxEigsIdxs)
                newExpoVector[cplxIdx] = flatProdExpoVector[i]
            end

            realP.tensors[axis, Tuple(newExpoVector)...] += 
            reduce(*, prodCoeff) * P.tensors[axis, expoVector...]

        end
    end

    # for the reduced dynamics f, need to take the product of the new polynomial array
    # with the inverse of the permutation matrix of conjugate pairs
    if invMultiply
        
        CL = zeros(Complex{Float64}, Nc + Nr, Nc + Nr)
        newPair = true

        for i in 1:P.domainDim
            if i in cplxEigsIdxs && newPair
                CL[i:i+1, i:i+1] = [1/2 -1im/2; 1/2 1im/2]
                newPair = false

            elseif i in cplxEigsIdxs && !newPair
                newPair = true

            elseif i in realEigsIdx
                CL[i, i] = 1
            end
        end

        coeffMatrix = zeros(ComplexF64, P.rangeDim, length(homogExponents))

        for axis in 1:P.rangeDim, (expoVectorIdx, expoVector) in enumerate(homogExponents)
            coeffMatrix[axis, expoVectorIdx] = realP.tensors[axis, expoVector...]
        end

        transformedCoeffs = inv(CL) * coeffMatrix

        for axis in 1:P.rangeDim, (expoVectorIdx, expoVector) in enumerate(homogExponents)

            realP.tensors[axis, expoVector...] = (transformedCoeffs[axis, expoVectorIdx])

        end
    end

    # remove the negligible imaginary residue
    if makeReal
        for axis in 1:P.rangeDim, expoVector in homogExponents
            realP.tensors[axis, expoVector...] = real(realP.tensors[axis, expoVector...])
        end
    end
    
    return realP

end

function realify(P::PolynomialArray, cplxEigsIdxs::Vector; 
    realEigsIdxs::Vector=[], invMultiply::Bool=false, 
    makeReal::Bool=false, returnArray::Bool=false)

    realP = PolynomialArray(P.order, P.domainDim, P.rangeDim)

    for k in 1:(P.order - 1)       
        update!(realP, 
                realify(P, homog_exponents(P.domainDim, k), cplxEigsIdxs, 
                realEigsIdxs=realEigsIdxs, 
                invMultiply=invMultiply, makeReal=makeReal),
                homog_exponents(P.domainDim, k))
    end

    if returnArray
        return SparseArray(real(realP.tensors));
    else
        return realP
    end
end


# Returns the sums along the left diagonals of a matrix
function left_diagonal_sums(matrix::Matrix{Complex{Float64}})
    nRows, nColumns = size(matrix)
    diagSums = zeros(ComplexF64, nColumns + nRows - 1)

    # loop over all left descendent diagonals of the matrix 
    # by first sweeping through the columns while staying in the first row of the matrix, 
    # and then start sweeping through the rows while staying in the last column. 
    # collect the sum of the elements in each of these diagonals and store them 
    # in vector diagSums
    
    for diag in 1:(nColumns + nRows - 1)
        startCol = min(nColumns, diag)
        startRow = max(0, diag - nColumns) + 1
        nElements = min(startCol, nRows - startRow + 1)

        for j in 0:nElements - 1
            diagSums[diag] += matrix[startRow + j, startCol - j]
        end
    end

    return diagSums
end

# find the Padé approximant of a polynomial given the orders of the
# numerator and denominator polynomials using the robust SVD method
function pade_approximant(P::Polynomial, N::Int64, M::Int64)

    # commented lines are expressions used by Kaszas and Haller (2025)

    maxOrder = N + M;

    if maxOrder > P.order
        error("Make sure the order of the polynomial is at least the sum of the 
        numerator and denominator orders")
    end

    bExponents = []
    for power in 0:(M)
        bExponents = vcat(bExponents, (homog_exponents(2, power)));
    end

    numberOfUnkns = length(bExponents);

    cExponents = [];
    for power in (N + 1):(N + M)
        #cExponents = vcat(cExponents, reverse(homog_exponents(2, power)));
        cExponents = vcat(cExponents, (homog_exponents(2, power)));
    end

    numberOfEqs = length(cExponents)

    C = zeros(ComplexF64, numberOfEqs, numberOfUnkns)
    for (eqIdx, expos) in enumerate(cExponents)

        α, β = expos .- 1;

        for k in 0:M, l in 0:k

            if β - l < 0 ||  α - k + l < 0
                continue
            else
                #colIndex = findfirst(==((l + 1, k - l + 1)), bExponents);
                colIndex = findfirst(==((k - l + 1, l + 1)), bExponents);
                C[eqIdx, colIndex] += P.tensor[α - k + l + 1, β - l + 1];
            end

        end

    end

    # find denominator b
    U, S, V = svd(C)
    zeroIdx = findfirst(<(1e-4), S)
    if zeroIdx !== nothing
        b = V[zeroIdx, :]
    else
        b = V[end, :]
    end

    #D = Diagonal(abs.(b) .+ sqrt(eps(Float64)))
    #Q, R = np.linalg.qr(transpose(C * D))
    #b = D * Q[:, end]
    #b /= norm(b)
    #b[abs.(b) .< 1e-12] .= 0.0

    # make new Polynomial for b
    bPoly = Polynomial(P.order, P.domainDim)
    for (bIdx, expo) in enumerate(bExponents)
        bPoly.tensor[expo...] = b[bIdx]
    end

    # find numerator a
    cExponents2 = [];
    for power in 0:N
        #cExponents2 = vcat(cExponents2, reverse(homog_exponents(2, power)));
        cExponents2 = vcat(cExponents2, (homog_exponents(2, power)));
    end

    aPoly = Polynomial(N, P.domainDim)
    for expos in (cExponents2)

        α, β = expos .- 1

        for k in 0:M, l in 0:k
            if β - l < 0 ||  α - k + l < 0
                continue
            else
                aPoly.tensor[α + 1, β + 1] += 
                P.tensor[α - k + l + 1, β - l + 1] * 
                bPoly.tensor[k - l + 1, l + 1];
                #bPoly.tensor[l + 1, k - l + 1];
            end
        end
    end

    return (aPoly, bPoly)

end

function pade_approximant(P::PolynomialArray, numOrder::Int64, denOrder::Int64)

    pades = Tuple{Polynomial, Polynomial}[]
    p = Polynomial(P.order, P.domainDim)

    for i in 1:P.rangeDim
        p.tensor = SparseArray(selectdim(P.tensors, 1, i))
        push!(pades, pade_approximant(p, numOrder, denOrder))
    end
    
    return pades
end