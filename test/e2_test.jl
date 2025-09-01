if sys2.linearSystem == []
    error("First find the linearized system!")
end

print("\nStarting parametrization.\n")

if sys2.computeFullSpectrum

    sys2.masterModes = masterModes
    slaveModes = setdiff(1:sys2.domainDim, masterModes)
    
    λ = sys2.eigenvalues
    λₗ = @view sys2.eigenvalues[masterModes]
    Λₗ = diagm(λₗ)

    P = sys2.eigenvectors
    L = @view P[:, masterModes]
    invP = inv(P)
    
else
    # if the spectrum is not fully computed, must provide master eigenmodes and eigenvalues        
    λₗ = masterEigenvalues
    Λₗ = diagm(λₗ)
    L = rightMasterEigenvectors
    X = leftMasterEigenvectors
    sys2.reducedDim = length(masterEigenvectors)

end

# initialize list of homogeneous exponent vectors with order 0 and 1 exponents
allHomogExponents = []
homogExponents0 = homog_exponents(sys2.reducedDim, 0)
homogExponents1 = homog_exponents(sys2.reducedDim, 1)
push!(allHomogExponents, homogExponents0)
push!(allHomogExponents, homogExponents1)

# initialize W polynomials
WPoly = PolynomialArray(maxOrder, sys2.reducedDim, sys2.domainDim)
for (expoVectorIdx, expoVector) in enumerate(homogExponents1), 
    rangeAxis in 1:sys2.domainDim

    WPoly.tensors[rangeAxis, expoVector...] = L[rangeAxis, expoVectorIdx]

end

# initialize DW polynomials
DWPoly = PolynomialArray(maxOrder - 1, sys2.reducedDim, sys2.domainDim*sys2.reducedDim)
update!(DWPoly, ∇(WPoly, allHomogExponents, 2), allHomogExponents[1])

# initialize f polynomials
fPoly = PolynomialArray(maxOrder, sys2.reducedDim, sys2.reducedDim)
for (expoVectorIdx, expoVector) in enumerate(homogExponents1),
    reducedAxis in 1:sys2.reducedDim

    fPoly.tensors[reducedAxis, expoVector...] = Λₗ[reducedAxis, expoVectorIdx]

end

# initialize DW*f polynomials
DWfPoly = PolynomialArray(maxOrder, sys2.reducedDim, sys2.domainDim)
*(DWPoly, fPoly, DWfPoly, allHomogExponents, 1)
updateDWfPoly = deepcopy(DWfPoly)


if sys2.BisTheIdentity == false
    # initialize B polynomials
    BPoly = PolynomialArray(maxOrder, sys2.reducedDim, sys2.domainDim * sys2.domainDim)

    for expoVector in homogExponents0, 
        matrixElementIdx in 1:sys2.domainDim * sys2.domainDim

        BPoly.tensors[matrixElementIdx, expoVector...] = sys2.B₀[matrixElementIdx]
            
    end

    # initialize B*DW*f polynomials
    BDWfPoly = PolynomialArray(maxOrder, sys2.reducedDim, sys2.domainDim)
end


# initialize polyTapes
fwd_poly_sweep!(sys2.dagF.polyTape, WPoly, allHomogExponents, 1)

# updateTape
updateTape = deepcopy(sys2.dagF.polyTape);

if sys2.BisTheIdentity == false
    fwd_poly_sweep!(sys2.dagB.polyTape, WPoly, allHomogExponents, 1)
    update!(BPoly, sys2.dagB.polyTape.data.y, allHomogExponents[2])
end

print("\nStarting main loop:\n")

k = 1

for k = 2:maxOrder - 1

    println("Computing order $k approximation...")

    # compute homogeneous exponents at order k in reducedDim variables
    homogExponents = homog_exponents(sys2.reducedDim, k)
    homogExpoList = enumerate(homogExponents)
    push!(allHomogExponents, homogExponents)

    # compute number of monomials at order k in reducedDim variables
    nMonomials = length(homogExponents)

    # sweep forward the DAG of F(W)
    print("Forward prop start... ")
    fwd_poly_sweep!(sys2.dagF.polyTape, WPoly, allHomogExponents, k)
    println("done")

    # the result is the polynomial in vector of coefficients form of [F(W)]ₖ
    E1 = homog_components(sys2.dagF.polyTape.data.y, homogExponents)

    # compute DWf
    *(DWPoly, fPoly, DWfPoly, allHomogExponents, k)

    # compute [B(W)DWf]ₖ and collect its polynomial in vector form
    # or do the same for [DWf]ₖ
    if sys2.BisTheIdentity
        E2 = homog_components(DWfPoly, homogExponents)
    else
        *(BPoly, DWfPoly, BDWfPoly, allHomogExponents, k)
        E2 = homog_components(BDWfPoly, homogExponents)
    end

    if sys2.computeFullSpectrum
        # solve system in modal coordinates
        η = - invP * sys2.invB₀ * (E1 - E2)
        ξ =  SparseArray(zeros(ComplexF64, sys2.domainDim, nMonomials))
        ϕ =  SparseArray(zeros(ComplexF64, sys2.domainDim, nMonomials))

        for (expoVectorIdx, expoVector) in homogExpoList

            eigSum = dot(expoVector .- 1, λₗ)

            for slaveIdx in slaveModes

                if abs(eigSum - λ[slaveIdx]) > sys2.crossResonanceTol
                    ξ[slaveIdx, expoVectorIdx] = 
                    η[slaveIdx, expoVectorIdx]/(λ[slaveIdx] - eigSum)
                else
                    error("Found cross resonance condition below tolerance")
                end
            end

            for masterIdx in masterModes
                
                if sys2.parametrizationStyle == "normal-form"

                    if abs(eigSum - λ[masterIdx]) > sys2.internalResonanceTol
                        ξ[masterIdx, expoVectorIdx] = 
                        η[masterIdx, expoVectorIdx]/(λ[masterIdx] - eigSum)
                    else
                        error("Found internal resonance condition below tolerance")
                    end

                elseif sys.parametrizationStyle == "normal-form-with-resonant-pair"

                    if (masterIdx == masterModes[1] && (expoVector[1] == expoVector[2] + 1))                           
                        
                        ϕ[1, expoVectorIdx] = -η[masterIdx, expoVectorIdx]
                    
                    elseif (masterIdx == masterModes[2] && (expoVector[2] == expoVector[1] + 1))

                        ϕ[2, expoVectorIdx] = -η[masterIdx, expoVectorIdx]

                    elseif abs(eigSum - λ[masterIdx]) > sys.internalResonanceTol

                        ξ[masterIdx, expoVectorIdx] = 
                        η[masterIdx, expoVectorIdx]/(λ[masterIdx] - eigSum)
                    else
                        error("Found internal resonance condition below tolerance")

                    end
                
                elseif sys2.parametrizationStyle == "graph"
                        ϕ[masterIdx, expoVectorIdx] = -η[masterIdx, expoVectorIdx]
                
                else 
                    error("Parametrization style not properly specified")
                end
            end
        end

        # add newly found Wₖ, fₖ and DWₖ to polynomial format
        update!(WPoly, P * ξ, homogExponents)
        update!(fPoly, ϕ, homogExponents)

        print("Update prop start... ")
        fwd_poly_sweep!(updateTape, WPoly, allHomogExponents, k)
        sys2.dagF.polyTape = deepcopy(updateTape)
        println("done")

    else 
        # solve system in physical coordinates
        Rₖ = - (E1 - E2)
        Wₖ = SparseArray(zeros(ComplexF64, sys2.domainDim, nMonomials))
        fₖ = SparseArray(zeros(ComplexF64, sys2.domainDim, nMonomials))

        for (expoVectorIdx, expoVector) in homogExpoList

            eigSum = dot(expoVector .- 1, λₗ)
            RHS = Rₖ[:, expoVectorIdx]
            LHS = eigSum * sys2.B₀ - sys2.Jacobian

            if sys2.parametrizationStyle == "normal-form"

                # check which modes belong to the resonant set of the monomial given by expoVector
                resonantSet = Int[]
                for eigIdx in eachindex(λₗ)
                    if abs(eigSum - λ[eigIdx]) < sys2.internalResonanceTol
                        push!(resonantSet, eigIdx)
                    end
                end

                if length(resonantSet) > 0
                    
                    println("Found internal resonance condition below tolerance, so cannot
                    perform full normal-form style parametrization.\nSwitching to mixed style...")
                    
                    bigLHS = SparseArray(zeros(ComplexF64, 
                                            sys2.domainDim + sys2.reducedDim, 
                                            sys2.domainDim + sys2.reducedDim))

                    bigLHS[1:sys2.domainDim - 1, 1:sys2.domainDim - 1] = LHS

                    bigLHS[1:sys2.domainDim - 1, sys2.domainDim:sys2.domainDim + length(resonantSet)] = 
                    sys2.B₀ * L[resonantSet]

                    bigLHS[sys2.domainDim:sys2.domainDim + length(resonantSet)] = 
                    X[resonantSet] * sys2.B₀

                    bigLHS[sys2.domainDim + length(resonantSet):end, sys2.domainDim + length(resonantSet):end] =
                    I(sys2.reducedDim - length(resonantSet))

                    bigRHS = SparseArray(zeros(ComplexF64, sys2.domainDim + sys2.reducedDim))
                    bigRHS[1:sys2.domainDim - 1] = RHS

                    bigSolution = inv(bigLHS) * bigRHS
                    Wₖ[:, expoVectorIdx] = bigSolution[1:sys2.domainDim - 1]
                    fₖ[resonantSet, expoVectorIdx] = bigSolution[sys2.domainDim:sys2.domainDim + length(resonantSet)]

                else
                    Wₖ[:, expoVectorIdx] = inv(LHS) * RHS

                end

            elseif sys2.parametrizationStyle == "graph"

                bigLHS = SparseArray(zeros(ComplexF64, 
                                        sys2.domainDim + sys2.reducedDim, 
                                        sys2.domainDim + sys2.reducedDim))
                                        
                bigLHS[1:sys2.domainDim - 1, 1:sys2.domainDim - 1] = LHS
                bigLHS[1:sys2.domainDim - 1, sys2.domainDim:sys2.domainDim + length(resonantSet)] = sys2.B₀ * L
                bigLHS[sys2.domainDim:end] = X * sys2.B₀

                bigRHS = SparseArray(zeros(ComplexF64, sys2.domainDim + sys2.reducedDim))
                bigRHS[1:sys2.domainDim - 1] = RHS

                bigSolution = inv(bigLHS) * bigRHS
                Wₖ[:, expoVectorIdx] = bigSolution[1:sys2.domainDim - 1]
                fₖ[:, expoVectorIdx] = bigSolution[sys2.domainDim:end]

            else
                error("Parametrization style not properly specified")
            end
        end

        # add newly found Wₖ, fₖ and DWₖ to polynomial format
        update!(WPoly, Wₖ, homogExponents)

        #print("Back prop start... ")
        #fwd_poly_sweep!(sys2.dagF.polyTape, WPoly, allHomogExponents, k)
        #println("done")

        #update!(fPoly, fₖ, homogExponents)
    end

    # update DW with the new, kth order W
    DWₖ = ∇(WPoly, allHomogExponents, k + 1)
    update!(DWPoly, DWₖ, allHomogExponents[k])
    *(DWPoly, fPoly, updateDWfPoly, allHomogExponents, k)
    DWfPoly = deepcopy(updateDWfPoly)

    # update B with the new, kth order W
    if sys2.BisTheIdentity == false
        fwd_poly_sweep!(sys2.dagB.polyTape, WPoly, allHomogExponents, k)
        update!(BPoly, sys2.dagB.polyTape.data.y, homogExponents)
    end

    println("Order $k done.\n")
end


sys2.W = WPoly
sys2.f = fPoly
print("\nParametrization complete.\n")