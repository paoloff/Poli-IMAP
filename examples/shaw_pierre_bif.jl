include("../src/Poli_IMap.jl");

# Constants
m = 1.0
c1 = 0.03
c2 = sqrt(3) * 0.03
k = 3.0
κ = 0.4
α = - 0.6
P = 3.0
ϵ = 0.003
Ω = 1.5
cte = 2.0

function F(x)
    return [x[3],
            
            x[4],
            
            - cte * (k / m) * x[1] + (k / m) * x[2] - (c1 + c2) / m * x[3] + (c2 / m) * x[4]
            - (κ / m) * x[1] * x[1] * x[1] - (α / m) * x[3] * x[3] * x[3]
            + ϵ * (P / m) * (x[5]),
            
            (k / m) * x[1] + (-cte * k / m) * x[2] + c2 / m * x[3] - ((c1 + c2) / m) * x[4],
            
            - 1.5 * x[6] - x[7] * x[6], # cos((1.5 + ΔΩ) * t)
            
            1.5 * x[5] + x[7] * x[5], # sin((1.5 + ΔΩ) * t)
            
            0.0 * x[7]] # ΔΩ
end

B = []

# system settings
domainDim = 7
reducedDim = 5
maxOrder = 9

# initializing and finding the linearized system
sys = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            F=F, B=B, 
            maxOrder=maxOrder);

initialize!(sys);
linearize!(sys, computeFullSpectrum=true);

eigvals = deepcopy(sys.eigenvalues);
eigvals[6] = eigvals[7];
eigvals[7] = 0 + 0.0im;
sys.eigenvalues = deepcopy(eigvals);
eigvecs = deepcopy(sys.rightEigenvectors);
eigvecs[:,6] = eigvecs[:,7];
eigvecs[:,7] = sys.rightEigenvectors[:,6];
sys.rightEigenvectors = deepcopy(eigvecs);
sys.leftEigenvectors = inv(sys.rightEigenvectors);

pSet = ParametrizationSettings(
        reducedDim = sys.reducedDim,
        λ = sys.eigenvalues,
        Y = sys.rightEigenvectors,
        X = sys.leftEigenvectors,
        tangAxes = [3, 4, 5, 6, 7],
        normAxes = [1, 2],
        autIdxs = [1, 2, 3, 4],
        nonAutIdxs = [5, 6],
        bifIdxs = [7],
        fullSpectrum = true,
        includesNonAutonomous = true,
        includesBifurcation = true,
        parametrizationStyle = "resonant",
        reducedDims = [2, 2, 1],
        homogExponents = Dict(),
        k₁ = 8,
        autDim = 4,
        autAxes = [1, 2, 3, 4],
        autTangAxes = [3, 4],
        autNormAxes = [1, 2],
        autReducedDim = 2,
        autReducedAxes = [1, 2],
        k₂ = 1,
        nonAutDim = 2,
        nonAutAxes = [5, 6],
        nonAutAxesPairs = [(5, 6)],
        nonAutReducedDim = 2,
        nonAutReducedAxes = [3, 4],
        nonAutReducedAxesMap = Dict((5, 6) => (3, 4)),
        k₃ = 1,
        bifDim = 1,
        bifReducedAxes = [5],
        bifReducedAxesMap = Dict(7 => 5),
        autAndNonAutAxes = [1, 2, 3, 4, 5, 6],
        autAndNonAutTangAxes = [3, 4, 5, 6],
        autAndNonAutNormAxes = [1, 2],
        autAndNonAutReducedAxes = [1, 2, 3, 4]
        );


# resonant condition function
function resonance_condition(exponents::Tuple, reducedAxis::Int64)

    resonant_pairs = Dict(1 => [1, 3], 2 => [2, 4], 3 => [3, 1], 4 => [4, 2])
    anti_resonant_pairs = Dict(1 => [2, 4], 2 => [1, 3], 3 => [2, 4], 4 => [1, 3])

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

parametrize_nonautonomous!(sys::System,
                        pSet::ParametrizationSettings,
                        pCache::ParametrizationCache);

parametrize_bifurcation!(sys::System, 
                        pSet::ParametrizationSettings,
                        pCache::ParametrizationCache);






