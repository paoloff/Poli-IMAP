if pCache.nonAutParametrizationDone == false
    error("First find the non-autonomous parametrization!")
end

# define the eigenvalues for homological equations
# λₜ = pSet.λ[pSet.autAndNonAutReducedAxes]
λₜ = pSet.λ[pSet.tangAxes]

# initialize polynomials for kAut = kNonAut = 0 and kBif = 1
bifHomogExponents = pSet.bifHomogExponents[2]

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
    bifExpo = ones(Int64, pSet.reducedDim)
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


print("\nStarting bifurcation parametrization.\n")

#for kBif in 1:pSet.k₃
kBif = 1
    
bifHomogExponents = pSet.bifHomogExponents[kBif + 1]

# for kNonAut in 0:pSet.k₂, kAut in 0:pSet.k₁

kNonAut = 0
kNonAut += 1
kAut = 0
kAut += 1

#if kNonAut == 0 && kAut == 0
#    continue
#end

k = [kAut, kNonAut, kBif]
combinedExponents = pSet.homogExponents[Tuple(k)]

# sweep forward the DAG of F(W)
# this is the most time consuming step in the algorithm
print("Forward prop start... ")
fwd_poly_sweep!(sys.dagF.polyTape, WPoly, inputExponents, k)
println("done")

E1 = homog_components(sys.dagF.polyTape.data.y, combinedExponents)

*(DWPoly, fPoly, DWfPoly, inputExponents, k)

if sys.BisTheIdentity
    E2 = homog_components(DWfPoly, combinedExponents)
else
    *(BPoly, DWfPoly, BDWfPoly, inputExponents, k)
    E2 = homog_components(BDWfPoly, combinedExponents)
end

E = - (E1 - E2)

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
                                λₜ, E, W₁, pSet.Yₜ, pSet.Xₜ,
                                WPoly, fPoly, pSet,
                                W₁=nothing)
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


println("\n Order $kBif done.\n\n")

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
