### Automatic differentiation functions

# simple addition
function Base.:+(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial,
    allHomogExponents::Vector, k::Int64)

    for expoVector in allHomogExponents[k + 1]
        YPoly.tensor[expoVector...] = 
        XPoly1.tensor[expoVector...] + XPoly2.tensor[expoVector...]
    end
end

function Base.:+(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial,
    allHomogExponents::Vector, k::Vector{Int64})

    for multiExpoVectors in Iterators.product([
        allHomogExponents[i][k[i] + 1] for i in eachindex(k)]...)

        expoSum = map(+, multiExpoVectors...) .- (length(k) - 1)

        YPoly.tensor[expoSum...] = XPoly1.tensor[expoSum...] + XPoly2.tensor[expoSum...]

    end

end


# simple subtraction
function Base.:-(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial,
    allHomogExponents::Vector, k::Int64)
    
    for expoVector in allHomogExponents[k + 1]
        YPoly.tensor[expoVector...] = 
        XPoly1.tensor[expoVector...] - XPoly2.tensor[expoVector...]
    end
end

function Base.:-(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial,
    allHomogExponents::Vector, k::Vector{Int64})
    
    for multiExpoVectors in Iterators.product([
        allHomogExponents[i][k[i] + 1] for i in eachindex(k)]...)

        expoSum = map(+, multiExpoVectors...) .- (length(k) - 1)

        YPoly.tensor[expoSum...] = XPoly1.tensor[expoSum...] - XPoly2.tensor[expoSum...]

    end
end


# simple multiplication
function Base.:*(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Int64)

    for j in 1:k + 1, expoVector1 in allHomogExponents[j], 
        expoVector2 in allHomogExponents[k - j + 2]
        
        YPoly.tensor[map(+, expoVector1, expoVector2 .- 1)...] += 
        XPoly1.tensor[expoVector1...] * XPoly2.tensor[expoVector2...]

    end

end

function Base.:*(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Vector{Int64})

    for j in Iterators.product([1:k[i] + 1 for i in eachindex(k)]...),
        multiExpoVectors1 in Iterators.product([
            allHomogExponents[i][j[i]] for i in eachindex(k)]...),
        multiExpoVectors2 in Iterators.product([
            allHomogExponents[i][k[i] - j[i] + 2] for i in eachindex(k)]...)

        expoSum1 = map(+, multiExpoVectors1...) .- (length(k) - 1)

        expoSum2 = map(+, multiExpoVectors2...) .- (length(k) - 1)

        YPoly.tensor[map(+, expoSum1, expoSum2) .- 1...] += 
        XPoly1.tensor[expoSum1...] * XPoly2.tensor[expoSum2...]

    end

end


# simple division
function Base.:/(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Int64)

    for j in 1:k, expoVector1 in allHomogExponents[j], 
        expoVector2 in allHomogExponents[k - j + 2]
        
        YPoly.tensor[map(+, expoVector1, expoVector2 .- 1)...] += 
        YPoly.tensor[expoVector1...] * XPoly2.tensor[expoVector2...]

    end

    for expoVector in allHomogExponents[k + 1]
        YPoly.tensor[expoVector...] = (-YPoly.tensor[expoVector...] + 
        XPoly1.tensor[expoVector...]) / XPoly2.tensor[ones(Int64, XPoly2.domainDim)...]
    end

end

# simple division
function Base.:/(XPoly1::Polynomial, XPoly2::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Vector{Int64})

    YPolyAux = Polynomial(XPoly1.order, XPoly1.domainDim);

    for j in Iterators.product([1:(k[i] + 1) for i in eachindex(k)]...),
        multiExpoVectors1 in Iterators.product([
            allHomogExponents[i][j[i]] for i in eachindex(k)]...),
            multiExpoVectors2 in Iterators.product([
                allHomogExponents[i][k[i] - j[i] + 2] for i in eachindex(k)]...)

                expoSum1 = map(+, multiExpoVectors1...) .- (length(k) - 1)

                expoSum2 = map(+, multiExpoVectors2...) .- (length(k) - 1)

                YPolyAux.tensor[map(+, expoSum1, expoSum2 .- 1)...] += 
                YPoly.tensor[expoSum1...] * XPoly2.tensor[expoSum2...]

    end
    
    for multiExpoVectors in Iterators.product([
        allHomogExponents[i][k[i] + 1] for i in eachindex(k)]...)

        expoSum = map(+, multiExpoVectors...) .- (length(k) - 1)

        YPoly.tensor[expoSum...] = (-YPolyAux.tensor[expoSum...] + 
        XPoly1.tensor[expoSum...]) / XPoly2.tensor[ones(Int64, XPoly2.domainDim)...]

    end

end


# product function used for AD of special functions 
function product!(XPoly1::Polynomial, XPoly2::Polynomial,
    YPoly::Polynomial,
    coeff_func::Function,
    allHomogExponents::Vector, 
    k::Int)
        
    for j in 1:k, expoVector1 in allHomogExponents[j], 
        expoVector2 in allHomogExponents[k - j + 2]

        YPoly.tensor[map(+, expoVector1, expoVector2 .- 1)...] += 
        coeff_func(j, k) * XPoly1.tensor[expoVector1...] * XPoly2.tensor[expoVector2...]
    end

end

function product!(XPoly1::Polynomial, XPoly2::Polynomial,
    YPoly::Polynomial,
    coeff_func::Function,
    allHomogExponents::Vector, 
    k::Vector{Int64})

    for j in Iterators.product([1:(k[i] + 1) for i in eachindex(k)]...),
        multiExpoVectors1 in Iterators.product([
            allHomogExponents[i][j[i]] for i in eachindex(k)]...),
        multiExpoVectors2 in Iterators.product([
            allHomogExponents[i][k[i] - j[i] + 2] for i in eachindex(k)]...)

        expoSum1 = map(+, multiExpoVectors1...) .- (length(k) - 1)

        expoSum2 = map(+, multiExpoVectors2...) .- (length(k) - 1)

        YPoly.tensor[map(+, expoSum1, expoSum2) .- 1...] += coeff_func(j, k) * 
        XPoly1.tensor[expoSum1...] * XPoly2.tensor[expoSum2...]

    end

end


### Special functions 

# sine
function Base.:sin(XPoly::Polynomial, YPoly::Vector{Polynomial}, 
    allHomogExponents::Vector, k::Int64)

    product!(YPoly[2], XPoly, YPoly[1], 
            (j, k) -> (k - j + 1.) / k, allHomogExponents, k)
end

function Base.:sin(XPoly::Polynomial, YPoly::Vector{Polynomial}, 
    allHomogExponents::Vector, k::Vector{Int64})

    product!(YPoly[2], XPoly, YPoly[1], 
            (j, k) -> (sum(k) - (sum(j) - length(k) + 1)  + 1.) / (sum(k)), allHomogExponents, k)
end

# cosine
function Base.:cos(XPoly::Polynomial, YPoly::Vector{Polynomial}, 
    allHomogExponents::Vector, k::Int64)

    product!(YPoly[1], XPoly, YPoly[2], 
            (j, k) -> -(k - j + 1.) / k, allHomogExponents, k)
end

function Base.:cos(XPoly::Polynomial, YPoly::Vector{Polynomial}, 
    allHomogExponents::Vector, k::Vector{Int64})

    product!(YPoly[1], XPoly, YPoly[2], 
            (j, k) -> -(sum(k) - (sum(j) - length(k) + 1) + 1.) / (sum(k)), allHomogExponents, k)
end


# exponential
function Base.:exp(XPoly::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Int64)

    product!(YPoly, XPoly, YPoly, 
            (j, k) -> (k - j + 1.) / k, allHomogExponents, k)
end

function Base.:exp(XPoly::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Vector{Int64})

    product!(YPoly, XPoly, YPoly, 
            (j, k) -> (sum(k) - (sum(j) - length(k) + 1) + 1.) / (sum(k)), allHomogExponents, k)
end

# natural logarithm
function Base.:log(XPoly::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Int64)

    product!(YPoly, XPoly, YPoly, 
            (j, k) -> j, allHomogExponents, k)

    for expoVector in allHomogExponents[k]
        YPoly.tensor[expoVector...] = (-YPoly.tensor[expoVector...]/k + 
        XPoly.tensor[expoVector...])/XPoly.tensor[1]
    end
end

function Base.:log(XPoly::Polynomial, YPoly::Polynomial, 
    allHomogExponents::Vector, k::Vector{Int64})

    product!(YPoly, XPoly, YPoly, 
            (j, k) -> j, allHomogExponents, k)

    for expoVector in allHomogExponents[k]
        YPoly.tensor[expoVector...] = (-YPoly.tensor[expoVector...]/k + 
        XPoly.tensor[expoVector...])/XPoly.tensor[1]
    end
end

# the AD formulae below are for vector or matrices of polynomials
# inner product between two 1D poly arrays
function dot!(XPoly1::PolynomialArray, XPoly2::PolynomialArray,
    YPoly::Polynomial, allHomogExponents::Vector, k::Int64)
    
    for j in 1:k + 1, expoVector1 in allHomogExponents[j], 
        expoVector2 in allHomogExponents[k - j + 2]

        for axis in XPoly1.domainDim
            YPoly.tensor[map(+, expoVector1, expoVector2 .- 1)...] += 
            XPoly1.tensors[axis, expoVector1] * XPoly2.tensors[axis, expoVector2]
        end
    end
end

function dot!(XPoly1::PolynomialArray, XPoly2::PolynomialArray,
    YPoly::Polynomial, allHomogExponents::Vector, k::Vector{Int64})


    for j in Iterators.product([1:k[i] + 1 for i in eachindex(k)]...),
        multiExpoVectors1 in Iterators.product([
            allHomogExponents[i][j[i]] for i in eachindex(k)]...),
        multiExpoVectors2 in Iterators.product([
            allHomogExponents[i][k[i] - j[i] + 2] for i in eachindex(k)]...),
        axis in XPoly1.domainDim

            expoSum1 = map(+, multiExpoVectors1...) .- (length(k) - 1)

            expoSum2 = map(+, multiExpoVectors2...) .- (length(k) - 1)

            YPoly.tensor[map(+, expoSum1, expoSum2) .- 1...] += XPoly1.tensor[axis, expoSum1...] * 
            XPoly2.tensor[axis, expoSum2...]
    end

end


# matrix-vector product between two poly arrays
function Base.:*(XPoly1::PolynomialArray, XPoly2::PolynomialArray, 
    YPoly::PolynomialArray, allHomogExponents::Vector, k::Int64)
    # here it is assumed that the first polynomial array represents an m x n matrix 
    # and the second represents an n x 1 vector

    newVecSize = floor(Int, XPoly1.rangeDim/XPoly2.rangeDim)

    for j in 1:k + 1, expoVector1 in allHomogExponents[j], 
        expoVector2 in allHomogExponents[k - j + 2]   

        for i = 1:newVecSize, axis in 1:XPoly2.rangeDim
            YPoly.tensors[i, map(+, expoVector1, expoVector2 .- 1)...] += 
            XPoly1.tensors[i + (axis - 1) * newVecSize, expoVector1...] *
            XPoly2.tensors[axis, expoVector2...]
        end
    end
end

function Base.:*(XPoly1::PolynomialArray, XPoly2::PolynomialArray, 
    YPoly::PolynomialArray, allHomogExponents::Vector, k::Vector{Int64})
    # here it is assumed that the first polynomial array represents an m x n matrix 
    # and the second represents an n x 1 vector

    newVecSize = floor(Int, XPoly1.rangeDim/XPoly2.rangeDim)

    for j in Iterators.product([1:k[i] + 1 for i in eachindex(k)]...),
        multiExpoVectors1 in Iterators.product([
            allHomogExponents[i][j[i]] for i in eachindex(k)]...),
        multiExpoVectors2 in Iterators.product([
            allHomogExponents[i][k[i] - j[i] + 2] for i in eachindex(k)]...),
        i in 1:newVecSize, axis in 1:XPoly2.rangeDim

        expoSum1 = map(+, multiExpoVectors1...) .- (length(k) - 1)

        expoSum2 = map(+, multiExpoVectors2...) .- (length(k) - 1)

        YPoly.tensors[i, map(+, expoSum1, expoSum2) .- 1...] += 
        XPoly1.tensors[i + (axis - 1) * newVecSize, expoSum1...] * 
        XPoly2.tensors[axis, expoSum2...]

    end
end