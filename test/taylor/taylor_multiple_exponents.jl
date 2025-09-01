include("../../src/Poli_IMap.jl");

#######################################################################################################
#######################################################################################################
#######################################################################################################

# examples of functions for testing

B = []

function F(x)
    return [sin((x[1]*x[2]-5.0))*x[1]*sin((x[2]-1.0)), cos((x[1]-1.0))]
end


function F(x)
    return [(cos(x[1]*sin(x[2]/(x[3]+3.0))))/(1.0-x[1]*x[3]+x[1]), cos((x[1]*x[2]-1.0)), 4.0*x[2]]
end

function F(x)
    return [1.0/(1.0-x[2]), cos((x[1]*x[2]-1.0))]
end


function F(x)

    ξ₁ = 0.2
    ξ₂ = 0.2
    μ = 1.4

    return [x[2],

            (cos(x[1] - x[3]) * (-1.0 * x[3] + x[1] - ξ₂ * x[4] + ξ₂ * x[2])
            + sin(x[1] - x[3]) * (x[2] * x[2] * cos(x[1] - x[3]) + x[4] * x[4] - μ)
            + 2.0 * x[1] - x[3] + (ξ₁ + ξ₂) * x[2] - ξ₂ * x[4])/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0),

            x[4],
            
            (sin(x[1] - x[3]) * (-1.0 * x[4] * x[4] * cos(x[1] - x[3]) + μ * cos(x[1] - x[3]) - 3.0 * x[2] * x[2])
            + cos(x[1] - x[3]) * (- 2.0 * x[1] + x[3] - (ξ₁ + ξ₂) * x[2] + ξ₂ * x[4])
            + 3.0 * x[3] - 3.0 * x[1] + 3.0 * ξ₂ * (x[4] - x[2]))/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0)]
end

#######################################################################################################
#######################################################################################################
#######################################################################################################

# taylor test for a single k

domainDim = 4
reducedDim = 4
rangeDim = 4
maxOrder = 12

sys = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            F=F, B=B, 
            maxOrder=maxOrder);

initialize!(sys);

allHomogExponents = []
homogExponents0 = homog_exponents(sys.reducedDim, 0)
homogExponents1 = homog_exponents(sys.reducedDim, 1)
push!(allHomogExponents, homogExponents0)
push!(allHomogExponents, homogExponents1)

L = [1.0 0.0 0.0 0.0; 0.0 1.0 0.0 0.0; 0.0 0.0 1.0 0; 0 0 0 1]

# initialize W polynomials
WPoly = PolynomialArray(maxOrder, sys.reducedDim, sys.domainDim)
for (expoVectorIdx, expoVector) in enumerate(homogExponents1), 
    rangeAxis in 1:sys.domainDim

    WPoly.tensors[rangeAxis, expoVector...] = L[rangeAxis, expoVectorIdx]

end

k = 1

fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, k)

# check
sys.dagF.polyTape

for i = 1:(maxOrder-2)
    k += 1

    # compute homogeneous exponents at order k in reducedDim variables
    homogExponents = homog_exponents(sys.reducedDim, k)
    push!(allHomogExponents, homogExponents)

    # compute number of monomials at order k in reducedDim variables
    nMonomials = length(homogExponents)

    # sweep forward the DAG of F(W)
    print("Forward prop start... ")
    fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, k)
    println("done")
end

# check
sys.dagF.polyTape.data.y[1]

#######################################################################################################
#######################################################################################################
#######################################################################################################

# taylor test for lists of k

domainDim = 3
reducedDim = 3
rangeDim = 3
maxOrder = 12

sys2 = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            F=F, B=B, 
            maxOrder=maxOrder);

initialize!(sys2);

pSet = ParametrizationSettings(
        reducedDim = sys2.reducedDim,
        tangAxes = [1, 2, 3],
        normAxes = [],

        fullSpectrum = true,
        includesNonAutonomous = true,
        parametrizationStyle = "normal-form",
        reducedDims = [2, 1],
        homogExponents = Dict(),

        k₁ = 12,
        autDim = 2,
        autIdxs = [1, 2],
        autAxes = [1, 2],
        autTangAxes = [1, 2],
        autNormAxes = [],
        autReducedDim = 2,
        autReducedAxes = [1, 2],

        k₂ = 12,
        nonAutDim = 1,
        nonAutIdxs = [3],
        nonAutAxes = [3],
        nonAutReducedDim = 1,
        nonAutReducedAxes = [3],

        autAndNonAutAxes = [1, 2, 3],
        autAndNonAutTangAxes = [1, 2, 3],
        autAndNonAutNormAxes = [],
        autAndNonAutReducedAxes = [1, 2, 3]
        );

generate_exponents!(pSet);


### Part I
inputExponents = pSet.autHomogExponents

L = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]

# initialize W polynomials
WPoly = PolynomialArray(maxOrder, sys2.reducedDim, sys2.domainDim)
for (expoVectorIdx, expoVector) in enumerate(inputExponents[2]), 
    rangeAxis in 1:sys2.domainDim

    WPoly.tensors[rangeAxis, expoVector...] = L[rangeAxis, expoVectorIdx]

end

for kAut = 0:(maxOrder-2)
    # sweep forward the DAG of F(W)
    print("Forward prop start... ")
    fwd_poly_sweep!(sys2.dagF.polyTape, WPoly, inputExponents, kAut + 1)
    println("done")
end

# check
sys2.dagF.polyTape.data.y[1]

### Part II
inputExponents = [pSet.autHomogExponents, pSet.nonAutHomogExponents];

L_up = [0.0; 0.0; 1];

# initialize W polynomials
for (expoVectorIdx, expoVector) in enumerate(inputExponents[2][2]),
    rangeAxis in 1:sys2.domainDim

    WPoly.tensors[rangeAxis, expoVector...] = L_up[rangeAxis, expoVectorIdx]

end

for kNonAut = 1:(pSet.k₂ - 1), kAut = 0:(pSet.k₁ - 1)

    # sweep forward the DAG of F(W)
    print("Forward prop start... ")
    fwd_poly_sweep!(sys2.dagF.polyTape, WPoly, inputExponents, [kAut, kNonAut])
    println("done")
end

# check
sys2.dagF.polyTape.data.y[1]