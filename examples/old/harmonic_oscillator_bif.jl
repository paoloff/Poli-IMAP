domainDim = 4
reducedDim = 4
rangeDim = 4
maxOrder = 10
matrixFormat = "full" 
computeFullSpectrum = true
parametrizationStyle = "normal-form"

omega = 0.6
A = 1

function F(x)

    return [x[2], 
            -1.0*x[1] -0.05*x[2] - 0.1*x[1]*x[1]*x[1],
            A * omega * x[4],   # sin(ωt)
            -A * omega*x[3]]    # cos(ωt)

end

B = []

################################################################################################
################################################################################################

using BifurcationKit, Plots

function Fsl(X, p)
    (;w, A) = p
    u, v, s, c = X
    u3 = u^3
    [v
    -u - 0.5 * v + 0.05 * u3 + A * s
    w * c + s * (1 - s^2 - c^2) # sin
    -w * s + c * (1 - s^2 - c^2) # cos
    ]
end

par_sl = (w = 0.01, A = 1)

u0 = zeros(4)
#u0[3] = 1
prob = BifurcationProblem(Fsl, u0, par_sl, (@optic _.w))

br = continuation(prob, PALC(), ContinuationPar(p_min=-5.0, p_max=5.0), bothside=true)

br_po = continuation(br, 2, ContinuationPar(),PeriodicOrbitOCollProblem(20, 5))

function Fsl(X, p)
    (;r, μ, ν, c3) = p 
    u, v = X
    ua = u^2 + v^2
    [
        r * u - ν * v - ua * (c3 * u - μ * v)
        r * v + ν * u - ua * (c3 * v + μ * u)
    ]
end

par_sl = (r = 0.1, μ = 0., ν = 1.0, c3 = 1.0)
u0 = zeros(2)
prob = BifurcationProblem(Fsl, u0, par_sl, (@optic _.r))
br = continuation(prob, PALC(), ContinuationPar(), bothside = true)

