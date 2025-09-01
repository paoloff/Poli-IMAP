include("../src/PoliMapSingle.jl");

domainDim = 4
reducedDim = 4
rangeDim = 4
maxOrder = 18
matrixFormat = "full" 
computeFullSpectrum = true
parametrizationStyle = "normal-form"

function F(x)
    return [sin((x[1]*x[2]-5.0))*x[1]/sin((x[2]-1.0)), cos((x[1]-1.0))]
end

ξ₁ = 0.2
ξ₂ = 0.2
μ = 1.4

function F(x)
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

B = []

sys = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            F=F, B=B, matrixFormat="full", 
            computeFullSpectrum=true, 
            parametrizationStyle="normal-form", 
            realify=true, maxOrder=18);

initialize!(sys);

allHomogExponents = []
homogExponents0 = homog_exponents(sys.reducedDim, 0)
homogExponents1 = homog_exponents(sys.reducedDim, 1)
push!(allHomogExponents, homogExponents0)
push!(allHomogExponents, homogExponents1)

L = [1.0 0.0 0 0; 0.0 1.0 0 0; 0 0 1 0; 0 0 0 1]
#L = [1.0 0; 0 1]

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

for i = 1:(maxOrder-1)
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
sys.dagF.polyTape.data.y
