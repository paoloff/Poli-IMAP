
# parametrize the invariant manifold up to order k₁ only for coordinates
# corresponding to the autonomous system
function parametrize_autonomous!(sys::System,
                                 pSet::ParametrizationSettings,
                                 pCache::ParametrizationCache)

    if sys.linearSystem == []
        error("First find the linearized system!")
    elseif pCache.fullParametrizationDone
        error("System already parametrized. Initialize a new one for a new parametrization.")
    end

    if pSet.fullSpectrum
        
        λₜ = pSet.λ[pSet.tangAxes]
        Λₜ = diagm(pSet.λ[pSet.tangAxes])
        Y = pSet.Y
        Yₜ = Y[:, pSet.autTangAxes]
        X = pSet.X
        
    else
        # if the spectrum is not fully computed, 
        # must provide the tangent eigenvectors and eigenvalues        
        λₜ = pSet.λₜ
        Λₜ = diagm(λₜ)
        Yₜ = pSet.Yₜ  
        Xₜ = pSet.Xₜ

    end

    allHomogExponents = []
    push!(allHomogExponents, pSet.autHomogExponents[1])
    push!(allHomogExponents, pSet.autHomogExponents[2])

    # initialize W polynomials
    WPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim)
    for (expoVectorIdx, expoVector) in enumerate(allHomogExponents[2]), 
        rangeAxes in 1:sys.domainDim

        WPoly.tensors[rangeAxes, expoVector...] = Yₜ[rangeAxes, expoVectorIdx]

    end

    # initialize DW polynomials
    DWPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim * sys.reducedDim)
    update!(DWPoly, ∇(WPoly, allHomogExponents, 1), allHomogExponents[1])

    # initialize f polynomials
    fPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.reducedDim)
    for (expoVectorIdx, expoVector) in enumerate(allHomogExponents[2]),
        reducedAxes in 1:sys.reducedDim
    
        fPoly.tensors[reducedAxes, expoVector...] = Λₜ[reducedAxes, expoVectorIdx]

    end
    
    # initialize DW*f polynomials
    DWfPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim)
    *(DWPoly, fPoly, DWfPoly, allHomogExponents, 1)
    updateDWfPoly = deepcopy(DWfPoly)

    if sys.BisTheIdentity == false
        # initialize B polynomials
        BPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim * sys.domainDim)

        for expoVector in homogExponents[1], 
            matrixElementIdx in 1:sys.domainDim * sys.domainDim

            BPoly.tensors[matrixElementIdx, expoVector...] = sys.B₀[matrixElementIdx]
                
        end

        # initialize B*DW*f polynomials
        BDWfPoly = PolynomialArray(sys.maxOrder, sys.reducedDim, sys.domainDim)
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

    print("\nStarting autonomous parametrization.\n")
    p = Progress(pSet.k₁ - 1; barglyphs=BarGlyphs("[=> ]"), barlen=20)

    for k = 2:(pSet.k₁ - 1)
    
        homogExponents = pSet.autHomogExponents[k + 1]
        push!(allHomogExponents, homogExponents)
       
        # sweep forward the DAG of F(W).
        # this is the most time consuming step in the algorithm
        #print("Forward prop start... ")
        fwd_poly_sweep!(sys.dagF.polyTape, WPoly, allHomogExponents, k)
        #println("done")

        # the result is the polynomial in vector of coefficients form of [F(W)]ₖ
        E1 = homog_components(sys.dagF.polyTape.data.y, homogExponents)

        # compute DWf
        *(DWPoly, fPoly, DWfPoly, allHomogExponents, k)

        # compute [B(W)DWf]ₖ and collect its polynomial in vector form
        # or do the same for [DWf]ₖ
        if sys.BisTheIdentity
            E2 = homog_components(DWfPoly, homogExponents)
        else
            *(BPoly, DWfPoly, BDWfPoly, allHomogExponents, k)
            E2 = homog_components(BDWfPoly, homogExponents)
        end

        # RHS residual from the lower order compositions
        E = (E1 - E2)

        # solve the homological equations for W and f
        if pSet.fullSpectrum
            solve_homological_equations!(sys, homogExponents,
                                        pSet.autNormAxes, 
                                        pSet.autTangAxes, 
                                        pSet.autReducedAxes, 
                                        λₜ, E, Y, X, 
                                        WPoly, fPoly, pSet)
        else
            solve_homological_equations!(sys, homogExponents, 
                                        pSet.autNormAxes, 
                                        pSet.autTangAxes, 
                                        pSet.autReducedAxes, 
                                        λₜ, E, Yₜ, Xₜ, 
                                        WPoly, fPoly, pSet)
        end

        # update prop
        #print("Update prop start... ")
        fwd_poly_sweep!(updateTapeF, WPoly, allHomogExponents, k)
        sys.dagF.polyTape = deepcopy(updateTapeF)
        #println("done")

        # update DW with the new, kth order W
        DWₖ = ∇(WPoly, allHomogExponents, k)
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

        #println("Order $k done.\n")
        next!(p)

    end

    pCache.WPoly = WPoly
    pCache.DWPoly = DWPoly
    pCache.fPoly = fPoly
    pCache.DWfPoly = DWfPoly

    if sys.BisTheIdentity == false
        pCache.BPoly = BPoly
        pCache.BDWfPoly = BDWfPoly
    end

    pCache.autParametrizationDone = true

    print("\nAutonomous parametrization complete.\n")


end