include("../src/Poli_IMAP.jl");

function critical_load(ξ::Float64) 
    μ_c = 7/2 - sqrt(2)
    return μ_c + (ξ * ξ) / 2 - ((3 - 2 * sqrt(2)) / 2) * 
    ((ξ - (4 + 5 * sqrt(2)) * ξ)^2) / ((ξ + ξ) * (ξ + 6 * ξ))
end

# dampings and load parameters values
ξ₁ = 0.2
ξ₂ = 0.2
μ = 0.55 * critical_load(ξ₁)

# vector field
function F(x)
    return [x[2],

            (cos(x[1] - x[3]) * (-1.0 * x[3] + x[1] - ξ₂ * x[4] + ξ₂ * x[2])
            + sin(x[1] - x[3]) * (x[2] * x[2] * cos(x[1] - x[3]) + x[4] * x[4] - μ)
            + 2.0 * x[1] - x[3] + (ξ₁ + ξ₂) * x[2] - ξ₂ * x[4])/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0),

            x[4],
            
            (sin(x[1] - x[3]) * (-1.0 * x[4] * x[4] * cos(x[1] - x[3]) + 
            μ * cos(x[1] - x[3]) - 3.0 * x[2] * x[2])
            + cos(x[1] - x[3]) * (- 2.0 * x[1] + x[3] - (ξ₁ + ξ₂) * x[2] + ξ₂ * x[4])
            + 3.0 * x[3] - 3.0 * x[1] + 3.0 * ξ₂ * (x[4] - x[2]))/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0)]
end

# mass matrix
B = [] 

# system settings
domainDim = 4
reducedDim = 2
maxOrder = 9

# initializing and finding the linearized system
sys = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            F=F, B=B, 
            maxOrder=maxOrder);

initialize!(sys);
linearize!(sys, computeFullSpectrum=true);

pSet = ParametrizationSettings(
        reducedDim = sys.reducedDim,
        λ = sys.eigenvalues,
        Y = sys.rightEigenvectors,
        X = sys.leftEigenvectors,
        tangAxes = [3, 4],
        normAxes = [1, 2],
        autIdxs = [1, 2, 3, 4],
        fullSpectrum = true,
        includesNonAutonomous = false,
        includesBifurcation = false,
        parametrizationStyle = "resonant",
        reducedDims = [2],
        homogExponents = Dict(),
        k₁ = maxOrder,
        autDim = 4,
        autAxes = [1, 2, 3, 4],
        autTangAxes = [3, 4],
        autNormAxes = [1, 2],
        autReducedDim = 2,
        autReducedAxes = [1, 2],
        autAndNonAutAxes = [1, 2, 3, 4],
        autAndNonAutTangAxes = [3, 4],
        autAndNonAutNormAxes = [1, 2],
        autAndNonAutReducedAxes = [1, 2]
        );


# resonant condition function
function resonance_condition(exponents::Tuple, reducedAxis::Int64)

    resonant_pairs = Dict(1 => [1], 2 => [2])
    anti_resonant_pairs = Dict(1 => [2], 2 => [1])

    if reducedAxis ∈ keys(resonant_pairs)

        if resonant_sum(exponents, 
                        resonant_pairs[reducedAxis], 
                        anti_resonant_pairs[reducedAxis],
                        startingSum=1) == 0

            return true
        else
            return false
        end
    end

    return false
    
end

# update parametrization settings
generate_exponents!(pSet);
generate_resonant_exponents!(pSet, resonance_condition);

# instantiate pCache
pCache = ParametrizationCache();

# parametrize
parametrize_autonomous!(sys::System,
                        pSet::ParametrizationSettings,
                        pCache::ParametrizationCache);