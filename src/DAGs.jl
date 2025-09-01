### DAG objects 

mutable struct DAG
    domainDim::Int64
    rangeDim::Int64 
    evalTape::Any
    polyTape::Any
end

#######################################################################################################
#######################################################################################################
#######################################################################################################

### DAG functions

# build a DAG object directly from a function
function build_dag(F::Function, 
                domainDim::Int64, 
                reducedDim::Int64, 
                rangeDim::Int64, 
                maxOrder::Int64)

    # build an empty scalar tape and populate the node data with values evaluated at x = 0
    if rangeDim isa Int64
        evalTape = record_tape(F, domainDim, rangeDim)

    else
        error("Range dimension must be an integer")

    end

    fwd_evaluation_sweep!(evalTape, zeros(domainDim))

    polyTape = record_poly_tape(evalTape, reducedDim, maxOrder)

    return DAG(domainDim, rangeDim, evalTape, polyTape)
    
end

#######################################################################################################
#######################################################################################################
#######################################################################################################

### Tape functions

# this function returns a PolynomialArray object with specified properties.
# first value in the poly series must be the value of the polys at x = 0
function poly_builder(domainDim::Int64,
                      referenceValue::Complex{Float64}, 
                      maxOrder::Int64,
                      operation::Symbol)
     
    zeroTuple = ones(Int64, domainDim)

    # if the operation is sine or cosine, 2 Polynomials must be stored in the node
    # and need to fill out the zero value of tensors appropriately
    if operation == :sin
        P1 = Polynomial(maxOrder, domainDim)
        P2 = Polynomial(maxOrder, domainDim)
        P1.tensor[zeroTuple...] = sin(referenceValue)
        P2.tensor[zeroTuple...] = cos(referenceValue)
        return [P1, P2], 2, 1

    elseif operation == :cos
        P1 = Polynomial(maxOrder, domainDim)
        P2 = Polynomial(maxOrder, domainDim)
        P1.tensor[zeroTuple...] = sin(referenceValue)
        P2.tensor[zeroTuple...] = cos(referenceValue)
        return [P1, P2], 2, 2
        
    # otherwise, only 1 Polynomial is enough
    else
        P1 = Polynomial(maxOrder, domainDim) 
        P1.tensor[zeroTuple...] = referenceValue
        return P1, 1, 1
    
    end
end


# create a tape storing the coefficients of the polynomial expansion of each node
function record_poly_tape(evalTape::ReverseAD.CompGraph{TapeData, NodeData}, 
    reducedDim::Int64, maxOrder::Int64)

    polyTape = CompGraph{PolynomialTapeData, PolynomialNodeData}(reducedDim, evalTape.rangeDim)

    for i in 1:(evalTape.rangeDim - 1)
        push!(polyTape.data.y, Polynomial())
    end

    # add new nodes to polyTape by looping through the evaluation tape nodes 
    # and fetching their data
    for node in evalTape.nodeList

        if node.operation == :sin || node.operation == :cos
            referenceValue = ComplexF64(evalTape.nodeList[node.parentIndices[1]].data.val)
        else
            referenceValue = ComplexF64(node.data.val)
        end

        poly, nOutputs, outputIdx = poly_builder(reducedDim, 
                                                referenceValue, 
                                                maxOrder, 
                                                node.operation)

        data = PolynomialNodeData(poly, nOutputs, node.constValue, outputIdx)

        newNode = GraphNode{PolynomialNodeData}(node.operation, 
                                                node.parentIndices, 
                                                node.constValue, 
                                                data)

        push!(polyTape.nodeList, newNode)
    end

    return polyTape 
end


# forward sweep through tape, to update a DAG with the next order expansion coefficients
# when the input is a polynomial
function fwd_poly_sweep!(tape::CompGraph{PolynomialTapeData, PolynomialNodeData}, 
    WPoly::PolynomialArray, allHomogExponents::Vector, k::Union{Int64, Vector{Int64}})
    
    # set tape
    tape.data.x = WPoly
    tape.data.iX = 1
    tape.data.iY = 1

    # sweep through all elements in the tape, calling fwd_poly_step! every time
    for node in tape.nodeList
        fwd_poly_step!(tape, node, allHomogExponents, k);
    end

end


# single node step of fwd_poly_sweep
function fwd_poly_step!(tape::CompGraph{PolynomialTapeData, PolynomialNodeData}, 
                        node::GraphNode{PolynomialNodeData},
                        allHomogExponents::Vector, 
                        k::Union{Int64, Vector{Int64}})
    
    op = node.operation
    nParents = length(node.parentIndices)

    if op == :const
        return
    end

    if nParents >= 1
        # check first parent node type
        if tape.nodeList[node.parentIndices[1]].data.nOutputs == 1
            input1 = tape.nodeList[node.parentIndices[1]].data.pol
        else
            input1 = tape.nodeList[node.parentIndices[1]].data.pol[
                tape.nodeList[node.parentIndices[1]].data.outputIdx]
        end
    end

    if nParents == 2
        # check second parent node type if it exists
        if tape.nodeList[node.parentIndices[2]].data.nOutputs == 1
            input2 = tape.nodeList[node.parentIndices[2]].data.pol
        else
            input2 = tape.nodeList[node.parentIndices[2]].data.pol[
                tape.nodeList[node.parentIndices[2]].data.outputIdx]
        end

        # evaluate operation and exit
        eval(op)(input1, input2, node.data.pol, allHomogExponents, k)
        return

    end

    # match operation type for specific instructions    
    # operations with 1 parent node
    if op == :sin || op == :cos
        eval(:sin)(input1, node.data.pol, allHomogExponents, k)
        eval(:cos)(input1, node.data.pol, allHomogExponents, k)

    elseif op != :input && op != :output
        eval(op)(input1, node.data.pol, allHomogExponents, k)
    
    # input
    elseif op == :input
        node.data.pol = Polynomial(node.data.pol.order, 
                                   node.data.pol.domainDim, 
                                   selectdim(tape.data.x.tensors, 1, tape.data.iX))
        tape.data.iX += 1
    
    # output
    else
        node.data.pol = input1
        tape.data.y[tape.data.iY] = node.data.pol
        tape.data.iY += 1
    
    end
    
end
