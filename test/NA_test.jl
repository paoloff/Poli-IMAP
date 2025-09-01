if sys.linearSystem == []
    error("First find the linearized system!")
end

print("\nStarting parametrization.\n")
autModes = [1, 2, 3, 4]
nonAutModes = [5, 6]
masterModes = [3, 4]
reducedModes = [3, 4, 5, 6]
reducedModesIdxsForUpdatingF = [1, 2, 3, 4]

if sys.computeFullSpectrum

    sys.masterModes = reducedModes
    slaveModes = setdiff(1:sys.domainDim, reducedModes)
    
    λ = sys.eigenvalues
    λₗ = @view sys.eigenvalues[reducedModes]
    Λₗ = diagm(λₗ)

    P = sys.eigenvectors
    autP = @view P[autModes, autModes]
    L = @view P[:, reducedModes]
    invP = inv(P)
    invAutP = inv(autP)
    
else
    # if the spectrum is not fully computed, must provide master eigenmodes and eigenvalues        
    λₗ = masterEigenvalues
    Λₗ = diagm(λₗ)
    L = rightMasterEigenvectors
    X = leftMasterEigenvectors
    sys.reducedDim = length(masterEigenvectors)

end

# initialize list of homogeneous exponent vectors with order 0 and 1 exponents
allHomogExponents = []
homogExponents0 = homog_exponents(sys.reducedDim, 0)
homogExponents1 = homog_exponents(sys.reducedDim, 1)
push!(allHomogExponents, homogExponents0)
push!(allHomogExponents, homogExponents1)
nMonomials = length(homogExponents1)

# autonomous exponents
autHomogExponents1 = []
autHomogExpoIdxs1 = []

# non-autonomous exponents of order 1
nonAutHomogExponents1 = []
nonAutHomogExpoIdxs1 = []

for (expoVectorIdx, expoVector) in enumerate(homogExponents1)

    if expoVector[sys.reducedDim - 1] == 1 && 
        expoVector[sys.reducedDim] == 1

        push!(autHomogExponents1, expoVector)
        push!(autHomogExpoIdxs1, expoVectorIdx)

    elseif expoVector[sys.reducedDim - 1] == 2  ||
        expoVector[sys.reducedDim] == 2
        
        push!(nonAutHomogExponents1, expoVector)
        push!(nonAutHomogExpoIdxs1, expoVectorIdx)
    end
end

# initialize W polynomials
WPoly = PolynomialArray(maxOrder, sys.reducedDim, sys.domainDim)
for (expoVectorIdx, expoVector) in zip(autHomogExpoIdxs1, autHomogExponents1), 
    rangeAxis in 1:sys.domainDim

    WPoly.tensors[rangeAxis, expoVector...] = L[rangeAxis, expoVectorIdx]

end

WPoly.tensors[5, 1, 1, 2, 1] = 0.5
WPoly.tensors[5, 1, 1, 1, 2] = 0.5

WPoly.tensors[6, 1, 1, 2, 1] = 0.5im
WPoly.tensors[6, 1, 1, 1, 2] = -0.5im


# initialize f polynomials
fPoly = PolynomialArray(maxOrder, sys.reducedDim, sys.reducedDim)
for (expoVectorIdx, expoVector) in enumerate(homogExponents1),
    reducedAxis in 1:sys.reducedDim

    fPoly.tensors[reducedAxis, expoVector...] = Λₗ[reducedAxis, expoVectorIdx]

end

# projected external force terms
forcing = sys.Jacobian[1:(sys.domainDim - 2), (sys.domainDim - 1):sys.domainDim]
forcing[3, 1] = 0.0045
forcing[3, 2] = 0.0045

# first order coefficients
ξ =  SparseArray(zeros(ComplexF64, 4, 4))
ϕ =  SparseArray(zeros(ComplexF64, 4, 4))
η = - invAutP * forcing

for (expoVectorIdx, expoVector) in zip(nonAutHomogExpoIdxs1, nonAutHomogExponents1)

    eigSum = dot(expoVector .- 1, λₗ)
    
    for slaveIdx in slaveModes

        ξ[slaveIdx, expoVectorIdx] = η[slaveIdx, expoVectorIdx - 2]/(λ[slaveIdx] - eigSum)
    end

    for (reducedIdx, masterIdx) in enumerate(masterModes)

        if reducedIdx == 1 && expoVectorIdx == 3
            ϕ[reducedIdx, expoVectorIdx] = -η[masterIdx, expoVectorIdx - 2]
        elseif reducedIdx == 2 && expoVectorIdx == 4
            ϕ[reducedIdx, expoVectorIdx] = -η[masterIdx, expoVectorIdx - 2]
        else
            ξ[masterIdx, expoVectorIdx] = η[masterIdx, expoVectorIdx - 2]/(λ[masterIdx] - eigSum)
        end

    end

end

update!(WPoly, autP * ξ, nonAutHomogExponents1, nonAutHomogExpoIdxs1, autModes, addToCurrent=true)
update!(fPoly, ϕ, nonAutHomogExponents1, nonAutHomogExpoIdxs1, reducedModesIdxsForUpdatingF, addToCurrent=true)

# initialize DW polynomials
DWPoly = PolynomialArray(maxOrder - 1, sys.reducedDim, sys.domainDim*sys.reducedDim)
update!(DWPoly, ∇(WPoly, allHomogExponents, 2), allHomogExponents[1])

# initialize DW*f polynomials
DWfPoly = PolynomialArray(maxOrder, sys.reducedDim, sys.domainDim)
*(DWPoly, fPoly, DWfPoly, allHomogExponents, 1)
updateDWfPoly = deepcopy(DWfPoly)

if sys.BisTheIdentity == false
    # initialize B polynomials
    BPoly = PolynomialArray(maxOrder, sys.reducedDim, sys.domainDim * sys.domainDim)

    for expoVector in homogExponents0, 
        matrixElementIdx in 1:sys.domainDim * sys.domainDim

        BPoly.tensors[matrixElementIdx, expoVector...] = sys.B₀[matrixElementIdx]
            
    end

    # initialize B*DW*f polynomials
    BDWfPoly = PolynomialArray(maxOrder, sys.reducedDim, sys.domainDim)
    *(BPoly, DWfPoly, BDWfPoly, allHomogExponents, 1)
    updateBDWfPoly = deepcopy(BDWfPoly)
end

# initialize polyTape
fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, 1)

# update tape for F
updateTapeF = deepcopy(sys.dagF.polyTape);

if sys.BisTheIdentity == false
    fwd_poly_sweep!(sys.dagB.polyTape, WPoly, allHomogExponents, 1)
    update!(BPoly, sys.dagB.polyTape.data.y, allHomogExponents[2])
    updateTapeB = deepcopy(sys.dagB.polyTape);
end

print("\nStarting main loop:\n")

k = 1
k += 1
println("Computing order $k approximation...")

# compute homogeneous exponents at order k in reducedDim variables
homogExponents = homog_exponents(sys.reducedDim, k)
homogExpoList = enumerate(homogExponents)
push!(allHomogExponents, homogExponents)

# compute number of monomials at order k in reducedDim variables
nMonomials = length(homogExponents)

# autonomous exponents
autHomogExponents = []
autHomogExpoIdxs = []

# non-autonomous exponents of order 1
nonAutHomogExponents = []
nonAutHomogExpoIdxs = []

for (expoVectorIdx, expoVector) in homogExpoList

    if expoVector[sys.reducedDim - 1] == 1 && expoVector[sys.reducedDim] == 1

        push!(autHomogExponents, expoVector)
        push!(autHomogExpoIdxs, expoVectorIdx)

    elseif expoVector[sys.reducedDim - 1] == 2 && expoVector[sys.reducedDim] == 1 ||
            expoVector[sys.reducedDim - 1] == 1 && expoVector[sys.reducedDim] == 2
        
        push!(nonAutHomogExponents, expoVector)
        push!(nonAutHomogExpoIdxs, expoVectorIdx)
    end
end

# sweep forward the DAG of F(W)
print("Forward prop start... ")
fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, k);
println("done")

# the result is the polynomial in vector of coefficients form of [F(W)]ₖ
E1 = homog_components(sys.dagF.polyTape.data.y, homogExponents)[1:sys.domainDim - 2,:]

# compute DWf
*(DWPoly, fPoly, DWfPoly, allHomogExponents, k);

# compute [B(W)DWf]ₖ and collect its polynomial in vector form
# or do the same for [DWf]ₖ
if sys.BisTheIdentity
    E2 = homog_components(DWfPoly, homogExponents)[1:sys.domainDim - 2,:]
else
    *(BPoly, DWfPoly, BDWfPoly, allHomogExponents, k)
    E2 = homog_components(BDWfPoly, homogExponents)[1:sys.domainDim - 2,:]
end

# solve system in modal coordinates
η = - invAutP * (E1 - E2)
ξ =  SparseArray(zeros(ComplexF64, 4, nMonomials))
ϕ =  SparseArray(zeros(ComplexF64, 4, nMonomials))

for (expoVectorIdx, expoVector) in zip(autHomogExpoIdxs, autHomogExponents)

    eigSum = dot(expoVector .- 1, λₗ)

    for slaveIdx in slaveModes

        ξ[slaveIdx, expoVectorIdx] = η[slaveIdx, expoVectorIdx]/(λ[slaveIdx] - eigSum)

    end

    for (reducedIdx, masterIdx) in enumerate(masterModes)
        
        if (masterIdx == masterModes[1] && (expoVector[1] == expoVector[2] + 1))                           
            
            ϕ[1, expoVectorIdx] = -η[masterIdx, expoVectorIdx]
        
        elseif (masterIdx == masterModes[2] && (expoVector[2] == expoVector[1] + 1))

            ϕ[2, expoVectorIdx] = -η[masterIdx, expoVectorIdx]

        elseif abs(eigSum - λ[masterIdx]) > sys.internalResonanceTol

            ξ[masterIdx, expoVectorIdx] = 
            η[masterIdx, expoVectorIdx]/(λ[masterIdx] - eigSum)
        else
            error("Resonance condition not accounted for, internal resonance might be present")

        end
    end
end

update!(WPoly, autP * ξ, autHomogExponents, autHomogExpoIdxs, autModes, addToCurrent=true)
update!(fPoly, ϕ, autHomogExponents, autHomogExpoIdxs, reducedModesIdxsForUpdatingF, addToCurrent=true)

# put a 1 in one of the non-autonomous coordinates and 0 in the other,
# then reverse the 1 and 0. Use f₁ and previously obtained P * ξ to compute E3

E3Poly = PolynomialArray(maxOrder, sys.reducedDim, 4)

for (expoVectorIdx, expoVector) in zip(nonAutHomogExpoIdxs, nonAutHomogExponents)

    if expoVector[3] == 2
        redAxis = 3
    else
        redAxis = 4
    end

    redExpoVector = ones(Int, sys.reducedDim)
    redExpoVector[redAxis] = 2

    for axis in 1:4
        for autAxis in 1:(sys.reducedDim - 2)   

            newExpoVector = collect(expoVector)
            newExpoVector[autAxis] += 1
            newExpoVector[redAxis] -= 1

            E3Poly.tensors[axis, expoVector...] +=  (newExpoVector[autAxis] - 1) * 
                                                WPoly.tensors[axis, newExpoVector...] * 
                                                fPoly.tensors[autAxis, redExpoVector...]
        end
    end
end

E3 = homog_components(E3Poly, homogExponents)

η += invAutP * E3 # check: is it = or += ?

for (expoVectorIdx, expoVector) in zip(nonAutHomogExpoIdxs, nonAutHomogExponents)

    eigSum = dot(expoVector .- 1, λₗ)
    
    for slaveIdx in slaveModes

        if abs(eigSum - λ[slaveIdx]) > sys.crossResonanceTol
            ξ[slaveIdx, expoVectorIdx] = 
            η[slaveIdx, expoVectorIdx]/(λ[slaveIdx] - eigSum)
        else
            error("Found cross resonance condition below tolerance")
        end
    end

    for (reducedIdx, masterIdx) in enumerate(masterModes)
        
        if masterIdx == masterModes[1] && (expoVector[1] == expoVector[2] + 2) && expoVector[4] == 2        
            
            ϕ[1, expoVectorIdx] = -η[masterIdx, expoVectorIdx]
        
        elseif masterIdx == masterModes[2] && (expoVector[2] == expoVector[1] + 2) && expoVector[3] == 2

            ϕ[2, expoVectorIdx] = -η[masterIdx, expoVectorIdx]

        elseif (expoVector[2] == expoVector[1]) && expoVector[3] == 2 &&  masterIdx == masterModes[1]
            ϕ[reducedIdx, expoVectorIdx] = -η[masterIdx, expoVectorIdx]

        elseif (expoVector[2] == expoVector[1]) && expoVector[4] == 2 &&  masterIdx == masterModes[2]
            ϕ[reducedIdx, expoVectorIdx] = -η[masterIdx, expoVectorIdx]

        elseif abs(eigSum - λ[masterIdx]) > sys.internalResonanceTol
            ξ[masterIdx, expoVectorIdx] = 
            η[masterIdx, expoVectorIdx]/(λ[masterIdx] - eigSum)

        else
            error("Resonance condition not accounted for, internal resonance might be present")

        end
    end

end

# add newly found Wₖ, fₖ and DWₖ to polynomial format
update!(WPoly, autP * ξ, nonAutHomogExponents, nonAutHomogExpoIdxs, autModes, addToCurrent=true)
update!(fPoly, ϕ, nonAutHomogExponents, nonAutHomogExpoIdxs, reducedModesIdxsForUpdatingF, addToCurrent=true)

# update prop
print("Update prop start... ")
fwd_poly_sweep!(updateTapeF, WPoly, allHomogExponents, k);
sys.dagF.polyTape = deepcopy(updateTapeF);
println("done")

# update DW with the new, kth order W
DWₖ = ∇(WPoly, allHomogExponents, k + 1);
update!(DWPoly, DWₖ, allHomogExponents[k]);
*(DWPoly, fPoly, updateDWfPoly, allHomogExponents, k);
DWfPoly = deepcopy(updateDWfPoly);

# update B with the new, kth order W
if sys.BisTheIdentity == false
    fwd_poly_sweep!(updateTapeB, WPoly, allHomogExponents, k)
    sys.dagB.polyTape = deepcopy(updateTapeB)
    update!(BPoly, sys.dagB.polyTape.data.y, homogExponents)
    *(BPoly, DWfPoly, updateBDWfPoly, allHomogExponents, k)
    BDWfPoly = deepcopy(updateBDWfPoly)
end

println("Order $k done.\n")


sys.W = WPoly
sys.f = fPoly
print("\nParametrization complete.\n")

