function parametrize!(sys)

    if sys.linearSystem == []
        error("First find the linearized system!")
    end

    print("\nStarting parametrization.\n")

    masterModes = [3,4] #[1, 2, 3, 4]
    slaveModes = setdiff(1:sys.domainDim, masterModes)
    λ = sys.eigenvalues
    λₗ = @view sys.eigenvalues[masterModes]
    Λₗ = diagm(λₗ)

    P = sys.eigenvectors
    L = @view P[:, masterModes]
    invP = inv(P)

    # initialize list of homogeneous exponent vectors with order 0 and 1 exponents
    allHomogExponents = []
    homogExponents0 = homog_exponents(sys.reducedDim, 0)
    homogExponents1 = homog_exponents(sys.reducedDim, 1)
    push!(allHomogExponents, homogExponents0)
    push!(allHomogExponents, homogExponents1)

    # initialize W polynomials
    WPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim)
    for (expoVectorIdx, expoVector) in enumerate(homogExponents1), 
        rangeAxis in 1:sys.domainDim

        WPoly.tensors[rangeAxis, expoVector...] = L[rangeAxis, expoVectorIdx]

    end

    # initialize DW polynomials
    DWPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim*sys.reducedDim)
    update!(DWPoly, ∇(WPoly, allHomogExponents, 2), allHomogExponents[1])

    # initialize f polynomials
    fPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.reducedDim)
    for (expoVectorIdx, expoVector) in enumerate(homogExponents1),
        reducedAxis in 1:sys.reducedDim

        fPoly.tensors[reducedAxis, expoVector...] = Λₗ[reducedAxis, expoVectorIdx]

    end

    # initialize DW*f polynomials
    DWfPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim)
    *(DWPoly, fPoly, DWfPoly, allHomogExponents, 1)
    updateDWfPoly = deepcopy(DWfPoly)

    if sys.BisTheIdentity == false
        # initialize B polynomials
        BPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim * sys.domainDim)

        for expoVector in homogExponents0, 
            matrixElementIdx in 1:sys.domainDim * sys.domainDim

            BPoly.tensors[matrixElementIdx, expoVector...] = sys.B₀[matrixElementIdx]
                
        end

        # initialize B*DW*f polynomials
        BDWfPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim)
        *(BPoly, DWfPoly, BDWfPoly, allHomogExponents, 1)
        updateBDWfPoly = deepcopy(BDWfPoly)

    end

    # initialize polyTapes
    fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, 1)

    # updateTape
    updateTapeF = deepcopy(sys.dagF.polyTape);

    if sys.BisTheIdentity == false
        fwd_poly_sweep!(sys.dagB.polyTape, WPoly, allHomogExponents, 1)
        update!(BPoly, sys.dagB.polyTape.data.y, allHomogExponents[2])
        updateTapeB = deepcopy(sys.dagB.polyTape);
    end

    print("\nStarting main loop:\n")


    for k = 2:sys.maxOrder - 1
        println("Computing order $k approximation...")

        # compute homogeneous exponents at order k in reducedDim variables
        homogExponents = homog_exponents(sys.reducedDim, k)
        homogExpoList = enumerate(homogExponents)
        push!(allHomogExponents, homogExponents)

        # compute number of monomials at order k in reducedDim variables
        nMonomials = length(homogExponents)

        # sweep forward the DAG of F(W)
        print("Forward prop start... ")
        fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, k)
        println("done")

        # the result is the polynomial in vector of coefficients form of [F(W)]ₖ
        E1 = homog_components(sys.dagF.polyTape.data.y, homogExponents)

        # compute DWf
        *(DWPoly, fPoly, DWfPoly, allHomogExponents, k)

        # compute [B(W)DWf]ₖ and collect its polynomial in vector form
        # or do the same for [DWf]ₖ
        if sys.BisTheIdentity
            E2 = homog_components(DWfPoly, homogExponents)
        else
            fwd_poly_sweep!(sys.dagB.polyTape, WPoly, allHomogExponents, k)
            update!(BPoly, sys.dagB.polyTape.data.y, homogExponents)
            *(BPoly, DWfPoly, BDWfPoly, allHomogExponents, k)
            E2 = homog_components(BDWfPoly, homogExponents)
        end

        if true
            # solve system in modal coordinates
            η = - invP * sys.invB₀ * (E1 - E2)
            ξ =  SparseArray(zeros(ComplexF64, sys.domainDim, nMonomials))
            ϕ =  SparseArray(zeros(ComplexF64, sys.domainDim, nMonomials))

            for (expoVectorIdx, expoVector) in homogExpoList

                eigSum = dot(expoVector .- 1, λₗ)

                for slaveIdx in slaveModes

                    if abs(eigSum - λ[slaveIdx]) > 0.01 #sys.crossResonanceTol
                        ξ[slaveIdx, expoVectorIdx] = 
                        η[slaveIdx, expoVectorIdx]/(λ[slaveIdx] - eigSum)
                    else
                        error("Found cross resonance condition below tolerance")
                    end
                end

                for masterIdx in masterModes
                    
                    if sys.parametrizationStyle == "normal-form"

                        if abs(eigSum - λ[masterIdx]) > sys.internalResonanceTol
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

                        #elseif (masterIdx == masterModes[3] && (expoVector[3] == expoVector[4] + 1))

                        #    ϕ[3, expoVectorIdx] = -η[masterIdx, expoVectorIdx]

                        #elseif (masterIdx == masterModes[4] && (expoVector[4] == expoVector[3] + 1))

                        #    ϕ[4, expoVectorIdx] = -η[masterIdx, expoVectorIdx]

                        elseif abs(eigSum - λ[masterIdx]) > 0.01 #sys.internalResonanceTol

                            ξ[masterIdx, expoVectorIdx] = 
                            η[masterIdx, expoVectorIdx]/(λ[masterIdx] - eigSum)
                        else
                            println(masterIdx)
                            println(expoVector)
                            error("Found internal resonance condition below tolerance")

                        end
                    
                    
                    elseif sys.parametrizationStyle == "graph"
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
            fwd_poly_sweep!(updateTapeF, WPoly, allHomogExponents, k)
            sys.dagF.polyTape = deepcopy(updateTapeF)
            println("done")

        else 
            # solve system in physical coordinates
            Rₖ = - (E1 - E2)
            Wₖ = SparseArray(zeros(ComplexF64, sys.domainDim, nMonomials))
            fₖ = SparseArray(zeros(ComplexF64, sys.domainDim, nMonomials))

            for (expoVectorIdx, expoVector) in homogExpoList

                eigSum = dot(expoVector .- 1, λₗ)
                RHS = Rₖ[:, expoVectorIdx]
                LHS = eigSum * sys.B₀ - sys.Jacobian

                if sys.parametrizationStyle == "normal-form"

                    # check which modes belong to the resonant set of the monomial given by expoVector
                    resonantSet = Int[]
                    for eigIdx in eachindex(λₗ)
                        if abs(eigSum - λ[eigIdx]) < sys.internalResonanceTol
                            push!(resonantSet, eigIdx)
                        end
                    end

                    if length(resonantSet) > 0
                        
                        println("Found internal resonance condition below tolerance, so cannot
                        perform full normal-form style parametrization.\nSwitching to mixed style...")
                        
                        bigLHS = SparseArray(zeros(ComplexF64, 
                                                sys.domainDim + sys.reducedDim, 
                                                sys.domainDim + sys.reducedDim))

                        bigLHS[1:sys.domainDim - 1, 1:sys.domainDim - 1] = LHS

                        bigLHS[1:sys.domainDim - 1, sys.domainDim:sys.domainDim + length(resonantSet)] = 
                        sys.B₀ * L[resonantSet]

                        bigLHS[sys.domainDim:sys.domainDim + length(resonantSet)] = 
                        X[resonantSet] * sys.B₀

                        bigLHS[sys.domainDim + length(resonantSet):end, sys.domainDim + length(resonantSet):end] =
                        I(sys.reducedDim - length(resonantSet))

                        bigRHS = SparseArray(zeros(ComplexF64, sys.domainDim + sys.reducedDim))
                        bigRHS[1:sys.domainDim - 1] = RHS

                        bigSolution = inv(bigLHS) * bigRHS
                        Wₖ[:, expoVectorIdx] = bigSolution[1:sys.domainDim - 1]
                        fₖ[resonantSet, expoVectorIdx] = bigSolution[sys.domainDim:sys.domainDim + length(resonantSet)]

                    else
                        Wₖ[:, expoVectorIdx] = inv(LHS) * RHS

                    end

                elseif sys.parametrizationStyle == "graph"

                    bigLHS = SparseArray(zeros(ComplexF64, 
                                            sys.domainDim + sys.reducedDim, 
                                            sys.domainDim + sys.reducedDim))
                                            
                    bigLHS[1:sys.domainDim - 1, 1:sys.domainDim - 1] = LHS
                    bigLHS[1:sys.domainDim - 1, sys.domainDim:sys.domainDim + length(resonantSet)] = sys.B₀ * L
                    bigLHS[sys.domainDim:end] = X * sys.B₀

                    bigRHS = SparseArray(zeros(ComplexF64, sys.domainDim + sys.reducedDim))
                    bigRHS[1:sys.domainDim - 1] = RHS

                    bigSolution = inv(bigLHS) * bigRHS
                    Wₖ[:, expoVectorIdx] = bigSolution[1:sys.domainDim - 1]
                    fₖ[:, expoVectorIdx] = bigSolution[sys.domainDim:end]

                else
                    error("Parametrization style not properly specified")
                end
            end

            # add newly found Wₖ, fₖ and DWₖ to polynomial format
            update!(WPoly, Wₖ, homogExponents)

            #print("Back prop start... ")
            #fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, k)
            #println("done")

            #update!(fPoly, fₖ, homogExponents)
        end

        # update DW with the new, kth order W
        DWₖ = ∇(WPoly, allHomogExponents, k + 1)
        update!(DWPoly, DWₖ, allHomogExponents[k])
        *(DWPoly, fPoly, updateDWfPoly, allHomogExponents, k)
        DWfPoly = deepcopy(updateDWfPoly)

        # update B with the new, kth order W
        if sys.BisTheIdentity == false
            fwd_poly_sweep!(updateTapeB, WPoly, allHomogExponents, k)
            sys.dagB.polyTape = deepcopy(updateTapeB)
            update!(BPoly, sys.dagB.polyTape.data.y, homogExponents)
            *(BPoly, DWfPoly, updateBDWfPoly, allHomogExponents, k)
            BDWfPoly = deepcopy(updateBDWfPoly)
        end

        println("Order $k done.\n")

    end

    sys.W = WPoly
    sys.f = fPoly
    print("\nParametrization complete.\n")
end