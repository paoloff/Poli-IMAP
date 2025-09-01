domainDim = 2
reducedDim = 1
rangeDim = 2
maxOrder = 10
matrixFormat = "full" 
computeFullSpectrum = true
parametrizationStyle = "normal-form"

function F(x)
    return [-1.0*x[1], -sqrt(24)*x[2] + x[1]*x[1] + x[1]*x[1]*x[1] + x[1]*x[1]*x[1]*x[1] + x[1]*x[1]*x[1]*x[1]*x[1]]
end


#=
function B(x)

    b = [0.0 for i in 1:4]
    nonZerosIdxs = [1, 4]

    for idx in nonZerosIdxs
        b[idx] = 1.0
    end

    return [bᵢ + 0.0*x[1] for bᵢ in b]
end=#


#=
function F(x)
    
    return [-1.0*x[1], 
            -sqrt(2)*x[2] + 1.0*x[1] - 1.0*x[1] - 4.0*x[2] + 5.0*x[1] 
            - x[1]*x[1] + x[1]*x[1]*x[1]
            + 5.0*x[1] - 2.3*x[2] + 5.3*x[1] 
            + x[1]*x[2]*x[1] 
            - 2.0*x[1] + 6.0*x[2],
            9.0*x[3],
            3.0*x[1] - 5.23*x[4]
            ]
end=#

