
if pCache.autParametrizationDone == false
    error("First find the autonomous parametrization!")
end

# define the eigenvalues for homological equations
λₜ = pSet.λ[pSet.tangAxes]

# initiate list of exponents
inputExponents = [pSet.autHomogExponents, pSet.nonAutHomogExponents]

# load cache
WPoly = deepcopy(pCache.WPoly)
DWPoly = deepcopy(pCache.DWPoly)
fPoly = deepcopy(pCache.fPoly)
DWfPoly = deepcopy(pCache.DWfPoly)
BPoly = deepcopy(pCache.DWfPoly)
BDWfPoly = deepcopy(pCache.BDWfPoly)

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
                                λₜ, E1, Y, X, WPoly, fPoly, pSet,
                                addToCurrentW=true,
                                addToCurrentf=true)
else
    solve_homological_equations!(sys, pSet.nonAutHomogExponents[2], 
                                pSet.autNormAxes, 
                                pSet.autTangAxes, 
                                pSet.autReducedAxes, 
                                λₜ, E1, Yₜ, Xₜ, WPoly, fPoly, pSet,
                                addToCurrentW=true,
                                addToCurrentf=true)
end

update!(DWPoly, ∇(WPoly, pSet.nonAutHomogExponents, 1), pSet.nonAutHomogExponents[1])
*(DWPoly, fPoly, DWfPoly, inputExponents, [0, 1])

updateDWfPoly = deepcopy(DWfPoly)
updateTapeF = deepcopy(sys.dagF.polyTape)

# no foward pass on dagB is necessary since B is fully autonomous
if sys.BisTheIdentity == false
    updateTapeB = deepcopy(sys.dagB.polyTape)
end

print("\nStarting non-autonomous parametrization.\n")

# general case 
#kNonAut = 0
#kNonAut += 1
#kNonAut in 1:pSet.k₂, 

#kAut = 0
#kAut += 1
#kAut in 0:pSet.k₁

for kNonAut = 1:pSet.k₂
    for kAut = 0:pSet.k₁

        if kNonAut == 1 && kAut == 0
            continue
        end

        k = [kAut, kNonAut]

        if pSet.includesBifurcation
            combinedExponents = pSet.homogExponents[(kAut, kNonAut, 0)]
        else
            combinedExponents = pSet.homogExponents[(kAut, kNonAut)]
        end

        # sweep forward the DAG of F(W)
        # this is the most time consuming step in the algorithm
        print("Forward prop start... ")
        fwd_poly_sweep!(sys.dagF.polyTape, WPoly, inputExponents, k)
        println("done")

        # the result is the polynomial in vector of coefficients form of [F(W)]ₖ
        E1 = homog_components(sys.dagF.polyTape.data.y, combinedExponents)

        # compute DWf
        *(DWPoly, fPoly, DWfPoly, inputExponents, k)

        # compute [B(W)DWf]ₖ and collect its polynomial in vector form
        # or do the same for [DWf]ₖ
        if sys.BisTheIdentity
            E2 = homog_components(DWfPoly, combinedExponents)
        else
            *(BPoly, DWfPoly, BDWfPoly, inputExponents, k)
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
            E3[pSet.autAxes, :] = sys.B₀[pSet.autAxes, pSet.autAxes] * homog_components(E3Poly, combinedExponents)
        else
            E3[pSet.autAxes, :] = homog_components(E3Poly, combinedExponents)
        end
            

        # RHS residual from the lower order compositions
        E = - (E1 - E2 - E3)

        # solve the homological equations for W and f
        # solve_homological_equations!(sys, homogExpoList, E, WPoly, fPoly)
        if pSet.fullSpectrum
            solve_homological_equations!(sys, combinedExponents, 
                                        pSet.autNormAxes, 
                                        pSet.autTangAxes, 
                                        pSet.autReducedAxes,
                                        λₜ, E, Y, X, WPoly, fPoly, pSet)
        else
            solve_homological_equations!(sys, combinedExponents, 
                                        pSet.autNormAxes, 
                                        pSet.autTangAxes, 
                                        pSet.autReducedAxes,
                                        λₜ, E, Yₜ, Xₜ, WPoly, fPoly, pSet)
        end

        # update prop
        print("Update prop start... ")
        fwd_poly_sweep!(updateTapeF, WPoly, inputExponents, k)
        sys.dagF.polyTape = deepcopy(updateTapeF)
        println("done")

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
    end

    println("\nOrder $kNonAut done.\n\n")

end


# pCache.allNonAutHomogExponents = allNonAutHomogExponents
pCache.WPoly = WPoly
pCache.DWPoly = DWPoly
pCache.fPoly = fPoly
pCache.DWfPoly = DWfPoly
pCache.BPoly = DWfPoly
pCache.BDWfPoly = BDWfPoly

pCache.nonAutParametrizationDone = true  

print("\nNon-autonomous parametrization complete.\n")

