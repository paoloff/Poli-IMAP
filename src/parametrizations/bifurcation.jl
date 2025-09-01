
# parametrize the invariant manifold up to order k₃ for the selected bifurcation coordinates.
# calling this function must be done AFTER calling parametrize_autonomous and parametrize_nonautonomous 

function parametrize_bifurcation!(sys::System, 
                                  pSet::ParametrizationSettings,
                                  pCache::ParametrizationCache)

    if pCache.nonAutParametrizationDone == false
        error("First find the non-autonomous parametrization!")
    end

    # define the eigenvalues for homological equations
    λₜ = pSet.λ[pSet.tangAxes]

    # initialize polynomials for kAut = kNonAut = 0 and kBif = 1
    bifHomogExponents = pSet.bifHomogExponents[1]

    WPoly = deepcopy(pCache.WPoly)
    DWPoly = deepcopy(pCache.DWPoly)
    fPoly = deepcopy(pCache.fPoly)
    DWfPoly = deepcopy(pCache.DWfPoly)
    updateDWfPoly = deepcopy(DWfPoly)
    updateTapeF = deepcopy(sys.dagF.polyTape)

    if sys.BisTheIdentity == false
        BPoly = deepcopy(pCache.BPoly)
        BDWfPoly = deepcopy(pCache.BDWfPoly)
    end

    for axis in pSet.bifIdxs
        bifExpo = ones(Int, pSet.reducedDim)
        bifExpo[pSet.bifReducedAxesMap[axis]] = 2
        WPoly.tensors[axis, bifExpo...] = 1.0
    end

    W₁ = homog_components(WPoly, homog_exponents(sys.reducedDim, 1))

    # no purely linear term is assumed for any bifurcation parameters
    # on the RHS of the system F(X).
    # however, there could be some purely linear terms in B(X)
    if sys.BisTheIdentity == false
        fwd_poly_sweep!(sys.dagB.polyTape, WPoly, bifHomogExponents, 1)
        update!(BPoly, sys.dagB.polyTape.data.y, bifHomogExponents)
        updateTapeB = deepcopy(sys.dagB.polyTape)
    end

    inputExponents = [pSet.autHomogExponents, 
                      pSet.nonAutHomogExponents, 
                      pSet.bifHomogExponents]

    
    print("\nStarting bifurcation parametrization.\n\n")

    for kBif in 1:pSet.k₃
        
        bifHomogExponents = pSet.bifHomogExponents[kBif + 1]
        p = Progress(pSet.k₂ * pSet.k₁ - 1; barglyphs=BarGlyphs("[=> ]"), barlen=20)

        for kNonAut in 0:pSet.k₂, kAut in 0:pSet.k₁

            if kNonAut == 0 && kAut == 0
                continue
            end

            k = [kAut, kNonAut, kBif]
            combinedExponents = pSet.homogExponents[Tuple(k)]

            # sweep forward the DAG of F(W)
            fwd_poly_sweep!(sys.dagF.polyTape, WPoly, inputExponents, k)

            E1 = homog_components(sys.dagF.polyTape.data.y, combinedExponents)

            *(DWPoly, fPoly, DWfPoly, inputExponents, k)

            if sys.BisTheIdentity
                E2 = homog_components(DWfPoly, combinedExponents)
            else
                *(BPoly, DWfPoly, BDWfPoly, inputExponents, k)
                E2 = homog_components(BDWfPoly, combinedExponents)
            end

            E = (E1 - E2)

            # solve the homological equations for W and f
            if pSet.fullSpectrum
                solve_homological_equations!(sys, combinedExponents,
                                            pSet.autAndNonAutNormAxes, 
                                            pSet.autAndNonAutTangAxes, 
                                            pSet.autAndNonAutReducedAxes, 
                                            λₜ, E, pSet.Y, pSet.X,
                                            WPoly, fPoly, pSet,
                                            W₁=W₁)
            else
                solve_homological_equations!(sys, combinedExponents,
                                            pSet.autAndNonAutNormAxes, 
                                            pSet.autAndNonAutTangAxes, 
                                            pSet.autAndNonAutReducedAxes, 
                                            λₜ, E, pSet.Yₜ, pSet.Xₜ,
                                            WPoly, fPoly, pSet,
                                            W₁=W₁)
            end

            # update prop
            fwd_poly_sweep!(updateTapeF, WPoly, inputExponents, k)
            sys.dagF.polyTape = deepcopy(updateTapeF)

            # update DW with the new, kth order W
            DWₖ = ∇(WPoly, inputExponents, k)
            update!(DWPoly, DWₖ, combinedExponents)
            *(DWPoly, fPoly, updateDWfPoly, inputExponents, k)
            DWfPoly = deepcopy(updateDWfPoly)
            
            # update B with the new, kth order W
            if sys.BisTheIdentity == false
                fwd_poly_sweep!(updateTapeB, WPoly, inputExponents, k)
                sys.dagB.polyTape = deepcopy(updateTapeB)
                update!(BPoly, sys.dagB.polyTape.data.y, combinedExponents)
                *(BPoly, DWfPoly, updateBDWfPoly, inputExponents, k)
                BDWfPoly = deepcopy(updateBDWfPoly)
            end

            next!(p, showvalues = [("Current order of bifurcation terms", kBif)])

        end

        #println("\n Order $kBif done.\n\n")
        next!(p)

    end

    print("\nBifurcation parametrization complete.\n")
    
    pCache.WPoly = WPoly
    pCache.DWPoly = DWPoly
    pCache.fPoly = fPoly
    pCache.DWfPoly = DWfPoly

    if sys.BisTheIdentity == false
        pCache.BPoly = BPoly
        pCache.BDWfPoly = BDWfPoly
    end
    
    pCache.bifParametrizationDone = true

end