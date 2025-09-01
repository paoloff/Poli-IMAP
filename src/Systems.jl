
### System objects

Base.@kwdef mutable struct System
    # Data fetched from input file
    domainDim::Int64
    reducedDim::Int64
    F::Function
    B::Any = []
    maxOrder::Int64

    # Data computed during linearization and parametrization
    initialized::Bool = false
    dagF::Any = [] 
    dagB::Any = [] 
    linearSystem::Any = [] 
    Jacobian::Any = []
    eigenvalues::Any = [] 
    rightEigenvectors::Any = [] 
    leftEigenvectors::Any = []
    B₀::Any = []
    invB₀::Any = [] 
    BisTheIdentity::Bool = false

    # Results
    W::Any = [] 
    WReal::Any = [] 
    f::Any = [] 
    fReal::Any = [] 

end

#######################################################################################################
#######################################################################################################
#######################################################################################################

### System functions

# initialize a system by building a dag
function initialize!(sys::System)

    print("Building DAG of F... ")
    sys.dagF = build_dag(sys.F, 
                        sys.domainDim, 
                        sys.reducedDim, 
                        sys.domainDim, 
                        sys.maxOrder)

    println("done")

    if sys.B isa Function
        print("Building DAG of B... ")
        sys.dagB = build_dag(sys.B, 
                            sys.domainDim, 
                            sys.reducedDim, 
                            sys.domainDim^2, 
                            sys.maxOrder)

        println("done\n")
        
    else
        print("Building DAG of B... ")
        println("B matrix not informed, setting it as the identity")
        sys.B = I(sys.domainDim)
        sys.B₀ = I(sys.domainDim)
        sys.invB₀ = I(sys.domainDim)
        sys.BisTheIdentity = true
        
    end
    
    sys.initialized = true

    return true
end


# compute spectrum given the linearized system
function compute_spectrum(linearSystemMatrix::Matrix{Float64}; useLAPackage::Bool)
    
    if useLAPackage
        eig = eigen(linearSystemMatrix)
        return eig.values, eig.vectors, inv(eig.vectors)
    else
        # need to provide alternative routine for computing the spectrum
        error("Missing alternative routine for computing the spectrum")
    end
end

    
# find the linerized system
function linearize!(sys::System;
                    computeFullSpectrum::Bool = true,
                    useLAPackage::Bool = true)

    if sys.initialized == false
        error("System not initialized!")
    end
    
    print("Computing Jacobian of F... ")
    J =  zeros(sys.domainDim, sys.domainDim)
    for i in 1:sys.domainDim
        yBar =  zeros(sys.domainDim)
        yBar[i] = 1.0
        _, Jcol = reverse_AD!(sys.dagF.evalTape, zeros(sys.domainDim), yBar)
        J[i, :] = Jcol
    end
    println("done")

    if sys.BisTheIdentity
        linearSystemMatrix = J

    else
        print("Computing B(0) and its inverse... ")
        sys.B₀ = sys.dagB.evalTape.data.y
        B₀InMatrixFormat =  reshape(sys.B₀, sys.domainDim, sys.domainDim)
        sys.invB₀ = inv(B₀InMatrixFormat)
        linearSystemMatrix = sys.invB₀ * J
        println("done")

    end

    if computeFullSpectrum
        print("Computing full spectrum... ")
        sys.eigenvalues, sys.rightEigenvectors, sys.leftEigenvectors = compute_spectrum(linearSystemMatrix, 
                                            useLAPackage = useLAPackage)
        println("done")

    end
    
    sys.linearSystem = linearSystemMatrix
    sys.Jacobian = J

    return true
end


# solve the homological equations given a list of homogeneous monomials.
# store the result back into WPoly and fPoly
function solve_homological_equations!(sys::System,
                                      homogExponents::Vector,
                                      normAxes::Vector{Int64},
                                      tangAxes::Vector{Int64},
                                      reducedAxes::Vector{Int64},
                                      λₜ::Vector{ComplexF64},
                                      E::SparseArray{ComplexF64},
                                      Y::Matrix{ComplexF64},
                                      X::Matrix{ComplexF64},
                                      WPoly::PolynomialArray,
                                      fPoly::PolynomialArray,
                                      pSet::ParametrizationSettings;
                                      W₁::Any = nothing,
                                      addToCurrentW::Bool = false,
                                      addToCurrentf::Bool = false)

    nMonomials = length(homogExponents)

    if pSet.fullSpectrum

        # solve system in modal coordinates
        η = - X * sys.invB₀ * E
        ξ =  SparseArray(zeros(ComplexF64, sys.domainDim, nMonomials))
        ϕ =  SparseArray(zeros(ComplexF64, sys.reducedDim, nMonomials))

        if pSet.parametrizationStyle == "normal-form"

            for (expoVectorIdx, expoVector) in enumerate(homogExponents)

                eigSum = dot(expoVector .- 1, λₜ)

                for normIdx in normAxes

                    if abs(eigSum - pSet.λ[normIdx]) > pSet.crossResonanceTol

                        ξ[normIdx, expoVectorIdx] = 
                        η[normIdx, expoVectorIdx]/(pSet.λ[normIdx] - eigSum)
                        
                    else
                        error("Found cross resonance condition below tolerance")
                    end
                end

                for tangIdx in tangAxes

                        if abs(eigSum - pSet.λ[tangIdx]) > pSet.internalResonanceTol

                            ξ[tangIdx, expoVectorIdx] = 
                            η[tangIdx, expoVectorIdx]/(pSet.λ[tangIdx] - eigSum)

                        else
                            error("Found internal resonance condition below tolerance")
                        end
                end
            end
                            
        elseif pSet.parametrizationStyle == "graph"

            if isnothing(W₁)

                for (expoVectorIdx, expoVector) in enumerate(homogExponents)

                    for normIdx in normAxes

                        if abs(eigSum - pSet.λ[normIdx]) > pSet.crossResonanceTol

                            ξ[normIdx, expoVectorIdx] = 
                            η[normIdx, expoVectorIdx]/(pSet.λ[normIdx] - eigSum)
                            
                        else
                            error("Found cross resonance condition below tolerance")
                        end
                    end

                    for (reducedIdx, tangIdx) in zip(reducedAxes, tangAxes)

                        ϕ[reducedIdx, expoVectorIdx] = -η[tangIdx, expoVectorIdx]

                    end
                end

            else
                @goto solve_full_system
            end
                
        elseif pSet.parametrizationStyle == "resonant"
                
            if isnothing(W₁)

                for (expoVectorIdx, expoVector) in enumerate(homogExponents)

                    eigSum = dot(expoVector .- 1, λₜ)

                    resonantSet = Int[]

                    for eigIdx in reducedAxes
                        if expoVector in pSet.resonantExponents[eigIdx]
                            push!(resonantSet, eigIdx)
                        end
                    end

                    for normIdx in normAxes

                        if abs(eigSum - pSet.λ[normIdx]) > pSet.crossResonanceTol

                            ξ[normIdx, expoVectorIdx] = 
                            η[normIdx, expoVectorIdx]/(pSet.λ[normIdx] - eigSum)
                            
                        else
                            error("Found cross resonance condition below tolerance")
                        end
                    end

                    for (reducedIdx, tangIdx) in zip(reducedAxes, tangAxes)

                        if reducedIdx ∉ resonantSet
                            if abs(eigSum - pSet.λ[tangIdx]) > pSet.internalResonanceTol

                                ξ[tangIdx, expoVectorIdx] = 
                                η[tangIdx, expoVectorIdx]/(pSet.λ[tangIdx] - eigSum)

                            else
                                error("Found internal resonance condition below tolerance")
                            end
                        else
                            ϕ[reducedIdx, expoVectorIdx] = -η[tangIdx, expoVectorIdx]
                        end

                    end

                end
                
            else
                @goto solve_full_system
            end

        end

        # add newly found Wₖ, fₖ and DWₖ to polynomial format
        update!(WPoly, Y * ξ, homogExponents, addToCurrent=addToCurrentW)
        update!(fPoly, ϕ, homogExponents, addToCurrent=addToCurrentf)

    else 
        # solve system in physical coordinates
        @label solve_full_system

        Wₖ = SparseArray(zeros(ComplexF64, sys.domainDim, nMonomials))
        fₖ = SparseArray(zeros(ComplexF64, sys.reducedDim, nMonomials))
        axes = vcat(normAxes, tangAxes)
        nAxes = length(axes)
        nReducedAxes = length(reducedAxes)

        if isnothing(W₁)
            Yₗ = Y[:, tangAxes]
            Xₗ = X[tangAxes, :]
        else
            Yₗ = W₁[:, reducedAxes]
            Xₗ = X[tangAxes, :]
        end

        for (expoVectorIdx, expoVector) in enumerate(homogExponents)

            eigSum = dot(expoVector .- 1, λₜ)
            RHS = E[axes, expoVectorIdx]
            LHS = (eigSum * sys.B₀ - sys.Jacobian)[axes, axes]

            if pSet.parametrizationStyle == "normal-form"

                # check which modes belong to the resonant set
                # of the monomial given by expoVector
                for eigIdx in tangAxes
                    if abs(eigSum - pSet.λ[eigIdx]) < pSet.internalResonanceTol
                        error("Found internal resonance condition below tolerance")
                    end
                end

                Wₖ[axes, expoVectorIdx] = inv(LHS) * RHS

            elseif pSet.parametrizationStyle == "resonant"
                
                resonantSet = Int[]

                for eigIdx in reducedAxes
                    if expoVector in pSet.resonantExponents[eigIdx]
                       push!(resonantSet, eigIdx)
                    end
                end

                if length(resonantSet) > 0
                    
                    bigLHS = SparseArray(zeros(ComplexF64, 
                                            nAxes + nReducedAxes, 
                                            nAxes + nReducedAxes))

                    bigLHS[1:nAxes, 1:nAxes] = LHS

                    bigLHS[1:nAxes, nAxes + 1:nAxes + length(resonantSet)] = 
                    sys.B₀[axes, axes] * Yₗ[axes, resonantSet]

                    bigLHS[nAxes + 1:nAxes + length(resonantSet), 1:nAxes] = 
                    Xₗ[resonantSet, axes] * sys.B₀[axes, axes]

                    bigLHS[nAxes + length(resonantSet) + 1:end, 
                    nAxes + length(resonantSet) + 1:end] =
                    I(nReducedAxes - length(resonantSet))

                    bigRHS = SparseArray(zeros(ComplexF64, nAxes + nReducedAxes))
                    bigRHS[1:nAxes] = RHS

                    bigSolution = inv(bigLHS) * bigRHS

                    Wₖ[axes, expoVectorIdx] = bigSolution[1:nAxes]

                    fₖ[resonantSet, expoVectorIdx] = bigSolution[nAxes + 1:nAxes + length(resonantSet)]

                else
                    Wₖ[axes, expoVectorIdx] = inv(LHS) * RHS
                end

            elseif pSet.parametrizationStyle == "graph"

                bigLHS = SparseArray(zeros(ComplexF64, 
                                        nAxes + nReducedAxes, 
                                        nAxes + nReducedAxes))
                                        
                bigLHS[1:nAxes, 1:nAxes] = LHS

                bigLHS[1:nAxes, nAxes + 1:nAxes + nReducedAxes] = 
                sys.B₀[axes, axes] * Yₗ[axes, :]

                bigLHS[nAxes + 1:end] = X[:, axes] * sys.B₀[axes, axes]

                bigRHS = SparseArray(zeros(ComplexF64, nAxes + nReducedAxes))
                bigRHS[1:nAxes] = RHS

                bigSolution = inv(bigLHS) * bigRHS
                Wₖ[axes, expoVectorIdx] = bigSolution[1:nAxes]
                fₖ[axes, expoVectorIdx] = bigSolution[nAxes + 1:end]

            else
                error("Parametrization style not properly specified")
            end
        end

        # add newly found Wₖ, fₖ and DWₖ to polynomial format
        update!(WPoly, Wₖ, homogExponents, addToCurrent=addToCurrentW)
        update!(fPoly, fₖ, homogExponents, addToCurrent=addToCurrentf)

    end

end
