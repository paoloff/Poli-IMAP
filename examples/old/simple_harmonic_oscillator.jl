domainDim = 6
reducedDim = 4
rangeDim = 6
maxOrder = 10
matrixFormat = "full" 
computeFullSpectrum = true
parametrizationStyle = "normal-form"


function F(x)
    return [x[2], 
            -1.0*x[1] -0.05*x[2] - 0.1*x[1]*x[1]*x[1],
            x[4],
            -1.7*x[3] -0.5*x[4] + 0.3*x[3]*x[3]*x[4],
            x[6],
            -10.2*x[5] -1.05*x[6] - 0.1*x[5]*x[5]*x[5]]
end

B = []