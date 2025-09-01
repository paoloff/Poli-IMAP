
# parametrize the invariant manifold up to order k₂ for the selected non-autonomous coordinates.
# calling function must be done AFTER calling parametrize_autonomous

function parametrize_nonautonomous!(sys::System, 
                                    pSet::ParametrizationSettings,
                                    pCache::ParametrizationCache)

    if pCache.autParametrizationDone == false
        error("First find the autonomous parametrization!")
    end

    # define the eigenvalues for homological equations
    λₜ = pSet.λ[pSet.tangAxes]

    # initialize polynomials for kAut = 0 and kNonAut = 1
    inputExponents = [pSet.autHomogExponents, pSet.nonAutHomogExponents]

    # load data from cache
    WPoly = deepcopy(pCache.WPoly)
    DWPoly = deepcopy(pCache.DWPoly)
    fPoly = deepcopy(pCache.fPoly)
    DWfPoly = deepcopy(pCache.DWfPoly)

    if sys.BisTheIdentity == false
        BPoly = deepcopy(pCache.BPoly)
        BDWfPoly = deepcopy(pCache.BDWfPoly)
    end

    for (axis1, axis2) in pSet.nonAutAxesPairs

        expoReal = ones(Int, pSet.reducedDim)
        expoReal[pSet.nonAutReducedAxesMap[(axis1, axis2)][1]] = 2

        WPoly.tensors[axis1, expoReal...] = 0.5
        WPoly.tensors[axis2, expoReal...] = 0.5im

        expoImag = ones(Int, pSet.reducedDim)
        expoImag[pSet.nonAutReducedAxesMap[(axis1, axis2)][2]] = 2

        WPoly.tensors[axis1, expoImag...] = 0.5
        WPoly.tensors[axis2, expoImag...] = - 0.5im

    end

    for (expoVector, axis, reducedAxis) in 
        zip(pSet.nonAutHomogExponents[2], pSet.nonAutAxes, pSet.nonAutReducedAxes)

        fPoly.tensors[reducedAxis, expoVector...] = pSet.λ[axis]

    end

    # compute forcing vector expression
    fwd_poly_sweep!(sys.dagF.polyTape, WPoly, inputExponents, [0, 1])
    E1 = homog_components(sys.dagF.polyTape.data.y, pSet.nonAutHomogExponents[2])
    
    # add forcing contribution to f and W
    if pSet.fullSpectrum
        solve_homological_equations!(sys, pSet.nonAutHomogExponents[2], 
                                    pSet.autNormAxes, 
                                    pSet.autTangAxes, 
                                    pSet.autReducedAxes, 
                                    λₜ, E1, pSet.Y, pSet.X, 
                                    WPoly, fPoly, pSet,
                                    addToCurrentW=true,
                                    addToCurrentf=true)
    else
        solve_homological_equations!(sys, pSet.nonAutHomogExponents[2], 
                                    pSet.autNormAxes, 
                                    pSet.autTangAxes, 
                                    pSet.autReducedAxes, 
                                    λₜ, E1, pSet.Yₜ, pSet.Xₜ, 
                                    WPoly, fPoly, pSet,
                                    addToCurrentW=true,
                                    addToCurrentf=true)
    end

    update!(DWPoly, ∇(WPoly, pSet.nonAutHomogExponents, 1), 
    pSet.nonAutHomogExponents[1])

    *(DWPoly, fPoly, DWfPoly, inputExponents, [0, 1])

    updateDWfPoly = deepcopy(DWfPoly)
    updateTapeF = deepcopy(sys.dagF.polyTape)

    # no foward pass on dagB is necessary since B is fully autonomous
    if sys.BisTheIdentity == false
        updateTapeB = deepcopy(sys.dagB.polyTape)
    end

    print("\nStarting non-autonomous parametrization.\n")
    
    # general case for  kNonAut = 1:k₂ and kAut = 0:k₁
    for kNonAut in 1:pSet.k₂

        p = Progress((pSet.k₁ + 1); barglyphs=BarGlyphs("[=> ]"), barlen=20)

        for kAut in 0:pSet.k₁

            if kNonAut == 1 && kAut == 0
                continue
            end

            if pSet.includesBifurcation
                combinedExponents = pSet.homogExponents[(kAut, kNonAut, 0)]
            else
                combinedExponents = pSet.homogExponents[(kAut, kNonAut)]
            end

            fwd_poly_sweep!(sys.dagF.polyTape, WPoly, inputExponents, [kAut, kNonAut])

            E1 = homog_components(sys.dagF.polyTape.data.y, combinedExponents)

            *(DWPoly, fPoly, DWfPoly, inputExponents, [kAut, kNonAut])

            if sys.BisTheIdentity
                E2 = homog_components(DWfPoly, combinedExponents)
            else
                *(BPoly, DWfPoly, BDWfPoly, inputExponents, [kAut, kNonAut])
                E2 = homog_components(BDWfPoly, combinedExponents)
            end

            E3Poly = PolynomialArray(sys.maxOrder, 
                                    pSet.reducedDim, 
                                    pSet.autDim)

            for expoVector in combinedExponents, 
                redAxes in pSet.nonAutReducedAxes
                    
                if expoVector[redAxes] == 1
                    continue
                end

                redExpoVector = ones(Int, pSet.reducedDim)
                redExpoVector[redAxes] = 2
        
                for axis in 1:pSet.autDim
                    for autAxes in pSet.autReducedAxes
        
                        newExpoVector = collect(expoVector)
                        newExpoVector[autAxes] += 1
                        newExpoVector[redAxes] -= 1

                        E3Poly.tensors[axis, expoVector...] +=  (newExpoVector[autAxes] - 1) * 
                                                            WPoly.tensors[axis, newExpoVector...] * 
                                                            fPoly.tensors[autAxes, redExpoVector...]
                    end
                end
            end

            E3 = SparseArray(zeros(ComplexF64, sys.domainDim, length(combinedExponents)))

            if sys.BisTheIdentity == false
                E3[pSet.autAxes, :] = sys.B₀[pSet.autAxes, pSet.autAxes] * 
                homog_components(E3Poly, combinedExponents)
            else
                E3[pSet.autAxes, :] = homog_components(E3Poly, combinedExponents)
            end
        
            E = (E1 - E2 - E3)

            if pSet.fullSpectrum
                solve_homological_equations!(sys, combinedExponents, 
                                            pSet.autNormAxes, 
                                            pSet.autTangAxes, 
                                            pSet.autReducedAxes,
                                            λₜ, E, pSet.Y, pSet.X, 
                                            WPoly, fPoly, pSet)
            else
                solve_homological_equations!(sys, combinedExponents, 
                                            pSet.autNormAxes, 
                                            pSet.autTangAxes, 
                                            pSet.autReducedAxes,
                                            λₜ, E, pSet.Yₜ, pSet.Xₜ, 
                                            WPoly, fPoly, pSet)
            end

            fwd_poly_sweep!(updateTapeF, WPoly, inputExponents, [kAut, kNonAut])
            sys.dagF.polyTape = deepcopy(updateTapeF)

            DWₖ = ∇(WPoly, inputExponents,[kAut, kNonAut])
            update!(DWPoly, DWₖ, combinedExponents)
            *(DWPoly, fPoly, updateDWfPoly, inputExponents, [kAut, kNonAut])
            DWfPoly = deepcopy(updateDWfPoly)
            
            if sys.BisTheIdentity == false
                fwd_poly_sweep!(updateTapeB, WPoly, inputExponents, [kAut, kNonAut])
                sys.dagB.polyTape = deepcopy(updateTapeB)
                update!(BPoly, sys.dagB.polyTape.data.y, combinedExponents)
                *(BPoly, DWfPoly, updateBDWfPoly, inputExponents, [kAut, kNonAut])
                BDWfPoly = deepcopy(updateBDWfPoly)
            end

            next!(p, showvalues = [("Current order of non-autonomous terms", kNonAut)])
        end

    end

    pCache.WPoly = WPoly
    pCache.DWPoly = DWPoly
    pCache.fPoly = fPoly
    pCache.DWfPoly = DWfPoly

    if sys.BisTheIdentity == false
        pCache.BPoly = BPoly
        pCache.BDWfPoly = BDWfPoly
    end

    pCache.nonAutParametrizationDone = true  

    print("Non-autonomous parametrization complete.\n")

end
