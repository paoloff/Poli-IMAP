Base.@kwdef mutable struct ParametrizationSettings
    
    # parametrization inputs
    reducedDim::Int64 = 0
    tangAxes::Vector{Int64} = [0]
    autIdxs::Vector{Int64} = [0]
    nonAutIdxs::Vector{Int64} = [0]
    bifIdxs::Vector{Int64} = [0]

    # essential settings
    fullSpectrum::Bool = true
    includesNonAutonomous::Bool = false
    includesBifurcation::Bool = false

    # things computed if fullSpectrum is true
    λ::Vector{ComplexF64} = [0.0im]
    Y::Matrix{ComplexF64} = [0.0im 0.0im]
    X::Matrix{ComplexF64} = [0.0im 0.0im]

    # inputs necessary if fullSpectrum is false
    λₜ::Vector{ComplexF64} = [0.0im]
    Yₜ::Matrix{ComplexF64} = [0.0im 0.0im]
    Xₜ::Matrix{ComplexF64} = [0.0im 0.0im]

    # style-related settings
    parametrizationStyle::String = "resonant" # "normal-form", "resonant", "graph"

    # tolerances
    crossResonanceTol::Float64 = 0.001
    internalResonanceTol::Float64 = 0.001

    # extra
    realify::Bool = false

    # constructed parameters
    normAxes::Vector{Int64} = [0]
    reducedDims::Vector{Int64} = [0]
    homogExponents::Dict = Dict()
    autHomogExponents::Vector = []
    nonAutHomogExponents::Vector = []
    bifHomogExponents::Vector = []


    # autonomous axis parameters
    k₁::Int64 = 0
    autDim::Int64 = 0
    autAxes::Vector{Int64} = [0]
    autTangAxes::Vector{Int64} = [0]
    autNormAxes::Vector{Int64} = [0]
    autTangEigvals::Vector{ComplexF64} = [0.0im]
    autReducedDim::Int64 = 0
    autReducedAxes::Vector{Int64} = [0]

    # non-autonomous axis parameters. 
    # all external drives are included in the reduced model 
    # i.e. as distinct coordinates in the tangent manifold
    k₂::Int64 = 0
    nonAutDim::Int64 = 0
    nonAutAxes::Vector{Int64} = [0]
    nonAutAxesPairs::Vector = []
    nonAutEigvals::Vector{ComplexF64} = [0.0im]
    nonAutReducedDim::Int64 = 0
    nonAutReducedAxes::Vector{Int64} = [0]
    nonAutReducedAxesMap::Dict = Dict()

    # bifurcation axis
    k₃::Int64 = 0
    bifDim::Int64 = 0
    bifReducedAxes::Vector{Int64} = [0]
    bifReducedAxesMap::Dict = Dict()

    # tangent and normal axis excluding the bifurcation parameters
    autAndNonAutAxes::Vector{Int64} = [0]
    autAndNonAutTangAxes::Vector{Int64} = [0]
    autAndNonAutNormAxes::Vector{Int64} = [0]
    autAndNonAutReducedAxes::Vector{Int64} = [0]

    # resonant monomials for the 'resonant' style of parametrization
    resonantExponents::Dict = Dict()
end


# generate all exponent combinations of monomials to use them later. 
# store them on pSet.homogExponents
function generate_exponents!(pSet::ParametrizationSettings)
    
    homogExponents = Dict()

    if pSet.includesBifurcation

        autHomogExponents = []
        nonAutHomogExponents = []
        bifHomogExponents = []

        for kAut in 0:pSet.k₁
            push!(autHomogExponents, homog_exponents(pSet.reducedDims, 
                                                     [kAut, 0, 0]))
        end

        for kNonAut in 0:pSet.k₂
            push!(nonAutHomogExponents, homog_exponents(pSet.reducedDims, 
                                                        [0, kNonAut, 0]))
        end

        for kBif in 0:pSet.k₃
            push!(bifHomogExponents, homog_exponents(pSet.reducedDims, 
                                                     [0, 0, kBif]))
        end

        for kAut in 0:pSet.k₁, kNonAut in 0:pSet.k₂, kBif in 0:pSet.k₃
            homogExponents[(kAut, kNonAut, kBif)] = homog_exponents(pSet.reducedDims, 
                                                    [kAut, kNonAut, kBif])
        end

        pSet.autHomogExponents = autHomogExponents
        pSet.nonAutHomogExponents = nonAutHomogExponents
        pSet.bifHomogExponents = bifHomogExponents

    elseif pSet.includesNonAutonomous

        autHomogExponents = []
        nonAutHomogExponents = []

        for kAut in 0:pSet.k₁
            push!(autHomogExponents, homog_exponents(pSet.reducedDims, 
                                                     [kAut, 0]))
        end

        for kNonAut in 0:pSet.k₂
            push!(nonAutHomogExponents, homog_exponents(pSet.reducedDims, 
                                                        [0, kNonAut]))
        end

        for kAut in 0:pSet.k₁, kNonAut in 0:pSet.k₂
            homogExponents[(kAut, kNonAut)] = homog_exponents(pSet.reducedDims, 
                                                              [kAut, kNonAut])
        end

        pSet.autHomogExponents = autHomogExponents
        pSet.nonAutHomogExponents = nonAutHomogExponents
    
    else
        autHomogExponents = []

        for kAut in 0:pSet.k₁
            push!(autHomogExponents, homog_exponents(pSet.reducedDims[1], kAut))
            homogExponents[kAut] = homog_exponents(pSet.reducedDims[1], kAut)
        end

        pSet.autHomogExponents = autHomogExponents

    end

    pSet.homogExponents = homogExponents

end


# selects the resonant exponents from pSet.homogExponents
# and store them on pSet.resonantExponents
function generate_resonant_exponents!(pSet::ParametrizationSettings,
                                      resonance_condition::Function)
    
    # resonantExponents = Dict([i => [] for i ∈ eachindex(pSet.autAndNonAutReducedAxes)])
    resonantExponents = Dict([i => [] for i in 1:pSet.reducedDim])

    for orders in keys(pSet.homogExponents), 
        exponent in pSet.homogExponents[orders],
        axis in pSet.autAndNonAutReducedAxes

        if resonance_condition(exponent, axis)
            push!(resonantExponents[axis], exponent)
        end
    end

    pSet.resonantExponents = resonantExponents
                    
end


# this function is useful when defining the resonance_condition function
function resonant_sum(exponents::Tuple,
                      resonantAxes::Vector{Int64}, 
                      antiResonantAxes::Any;
                      startingSum::Int64,
                      weights::Vector{Int64}=[0])
    
    sum = startingSum

    if length(weights) == 1
        weights = ones(Int64, length(exponents))
    end

    if isnothing(antiResonantAxes)
        for i in eachindex(exponents)
            if i ∈ resonantAxes
                sum += - weights[i] * (exponents[i] - 1)
            end
        end

    else
        for i in eachindex(exponents)
            if i ∈ resonantAxes
                sum -= weights[i] * (exponents[i] - 1)
            elseif i ∈ antiResonantAxes
                sum += weights[i] * (exponents[i] - 1)
            end
        end
    end

    return sum

end

