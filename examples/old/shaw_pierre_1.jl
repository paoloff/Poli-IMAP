domainDim = 6
reducedDim = 4
rangeDim = 6
maxOrder = 10
matrixFormat = "full" 
computeFullSpectrum = true
parametrizationStyle = "normal-form-with-resonant-pair"

m = 1.0 
c1 = 0.03
c2 = sqrt(3) * 0.03
k = 3.0
κ = 0.4
α = - 0.6
P = 3.0
ϵ = 0.003
Ω = 1.5
cte = 2.0

function F(x)
    return [x[3],
            
            x[4],
            
            - cte * (k / m) * x[1] + (k / m) * x[2] - (c1 + c2) / m * x[3] + (c2 / m) * x[4]
            - (κ / m) * x[1] * x[1] * x[1] - (α / m) * x[3] * x[3] * x[3]
            + ϵ * (P / m) * (x[5]),
            
            (k / m) * x[1] + (-cte * k / m) * x[2] + c2 / m * x[3] - ((c1 + c2) / m) * x[4],
            
            - Ω * x[6], # cos(Ωt)
            
            Ω * x[5]] # sin(Ωt)
end


B = []
