include("../src/Poli_IMAP.jl");

function critical_load(ξ::Float64) 
    μ_c = 7/2 - sqrt(2)
    return μ_c + (ξ * ξ) / 2 - ((3 - 2 * sqrt(2)) / 2) * 
    ((ξ - (4 + 5 * sqrt(2)) * ξ)^2) / ((ξ + ξ) * (ξ + 6 * ξ))
end

# dampings and load parameters values
ξ₁ = 0.1
ξ₂ = 0.1
μ = 0.4408 #0.55 * critical_load(ξ₁) # static load
Δμ = 0.6 #0.2 * critical_load(ξ₁) # amplitude of pulsating load
Ω = 2 * 0.4162 # near parametric resonance

# vector field
function F(x)
    return [x[2],

            (cos(x[1] - x[3]) * (-1.0 * x[3] + x[1] - ξ₂ * x[4] + ξ₂ * x[2])
            + sin(x[1] - x[3]) * (x[2] * x[2] * cos(x[1] - x[3]) + x[4] * x[4] - μ - Δμ * x[5])
            + 2.0 * x[1] - x[3] + (ξ₁ + ξ₂) * x[2] - ξ₂ * x[4])/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0),

            x[4],
            
            (sin(x[1] - x[3]) * (-1.0 * x[4] * x[4] * cos(x[1] - x[3]) + 
            (μ + Δμ * x[5]) * cos(x[1] - x[3]) - 3.0 * x[2] * x[2])
            + cos(x[1] - x[3]) * (- 2.0 * x[1] + x[3] - (ξ₁ + ξ₂) * x[2] + ξ₂ * x[4])
            + 3.0 * x[3] - 3.0 * x[1] + 3.0 * ξ₂ * (x[4] - x[2]))/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0),

            - Ω * x[6],
            
            Ω * x[5]]
end


# mass matrix
B = [] 

# system settings
domainDim = 6
reducedDim = 4
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
        tangAxes = [3, 4, 5, 6],
        normAxes = [1, 2],
        autIdxs = [1, 2, 3, 4],
        nonAutIdxs = [5, 6],
        fullSpectrum = true,
        includesNonAutonomous = true,
        parametrizationStyle = "resonant",
        reducedDims = [2, 2],
        homogExponents = Dict(),
        k₁ = 8,
        autDim = 4,
        autAxes = [1, 2, 3, 4],
        autTangAxes = [3, 4],
        autNormAxes = [1, 2],
        autReducedDim = 2,
        autReducedAxes = [1, 2],
        k₂ = 4,
        nonAutDim = 2,
        nonAutAxes = [5, 6],
        nonAutAxesPairs = [(5, 6)],
        nonAutReducedDim = 2,
        nonAutReducedAxes = [3, 4],
        nonAutReducedAxesMap = Dict((5, 6) => (3, 4)),
        autAndNonAutAxes = [1, 2, 3, 4, 5, 6],
        autAndNonAutTangAxes = [3, 4, 5, 6],
        autAndNonAutNormAxes = [1, 2],
        autAndNonAutReducedAxes = [1, 2, 3, 4]
        );


# resonant condition function
function resonance_condition(exponents::Tuple, reducedAxis::Int64)

    # primary resonances
    resonant_pairs = Dict(1 => [1, 3], 2 => [2, 4], 3 => [3], 4 => [4])
    anti_resonant_pairs = Dict(1 => [2, 4], 2 => [1, 3], 3 => [4], 4 => [3])

    if reducedAxis ∈ keys(resonant_pairs)

        if resonant_sum(exponents, 
                        resonant_pairs[reducedAxis], 
                        anti_resonant_pairs[reducedAxis],
                        startingSum=1,
                        weights=[1, 1, 2, 2]) == 0

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

# include all linear parametric monomials in the reduced equations
push!(pSet.resonantExponents[1],(1, 2, 1, 2));
push!(pSet.resonantExponents[1],(2, 1, 1, 2));
push!(pSet.resonantExponents[1],(2, 1, 2, 1));
push!(pSet.resonantExponents[2],(1, 2, 1, 2));
push!(pSet.resonantExponents[2],(1, 2, 2, 1));
push!(pSet.resonantExponents[2],(2, 1, 2, 1));

# instantiate pCache
pCache = ParametrizationCache();

# parametrize
parametrize_autonomous!(sys::System,
                        pSet::ParametrizationSettings,
                        pCache::ParametrizationCache);

parametrize_nonautonomous!(sys::System,
                        pSet::ParametrizationSettings,
                        pCache::ParametrizationCache);

fPoly = pCache.fPoly;
WPoly = pCache.WPoly;

WArray = realify(WPoly, [1,2,3,4], makeReal=true, returnArray=true);
fArray = realify(fPoly, [1,2,3,4], invMultiply=true, makeReal=true, returnArray=true);

using MAT
path = "/Users/paolofurlanettoferrari/Documents/MATLAB/MatCont/duffing & more/roms/"
matwrite(path * "f.mat", Dict("f" => Array(fArray)));
matwrite(path * "W.mat", Dict("W" => Array(WArray)));


