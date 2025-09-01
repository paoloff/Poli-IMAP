
domainDim = 100
reducedDim = 6
rangeDim = 100
maxOrder = 7
matrixFormat = "full" 
computeFullSpectrum = true
parametrizationStyle = "normal-form"

function F(x)
    #A = zeros(100)

    B = [0.0 for i in 1:100]
    a = 1.0
    b = 0.5
    c = 0.2

    #for i = 1:2:99
    #    a, b, c = rand(3)
    #    A[i] = x[i+1]
    #    A[i+1] = -a*x[i] -(b/10)*x[i+1] - (c/10)*x[i]*x[i]*x[i]
    #end

    return [(-rand(1)[1]*x[i] -(b/10)*x[i]*x[1] - (c/10)*x[i]*x[i]*x[i]) for i in 1:100]
end

B = []

