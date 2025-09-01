include("../src/Poli_IMAP.jl");

function critical_load(ξ::Float64) 
    μ_c = 7/2 - sqrt(2)
    return μ_c + (ξ * ξ) / 2 - ((3 - 2 * sqrt(2)) / 2) * 
    ((ξ - (4 + 5 * sqrt(2)) * ξ)^2) / ((ξ + ξ) * (ξ + 6 * ξ))
end

# dampings and load parameters values
ξ₁ = 0.2
ξ₂ = 0.2
μ = 0.55 * critical_load(ξ₁)

# vector field
function F(x)
    return [x[2],

            (cos(x[1] - x[3]) * (-1.0 * x[3] + x[1] - ξ₂ * x[4] + ξ₂ * x[2])
            + sin(x[1] - x[3]) * (x[2] * x[2] * cos(x[1] - x[3]) + x[4] * x[4] - μ)
            + 2.0 * x[1] - x[3] + (ξ₁ + ξ₂) * x[2] - ξ₂ * x[4])/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0),

            x[4],
            
            (sin(x[1] - x[3]) * (-1.0 * x[4] * x[4] * cos(x[1] - x[3]) + 
            μ * cos(x[1] - x[3]) - 3.0 * x[2] * x[2])
            + cos(x[1] - x[3]) * (- 2.0 * x[1] + x[3] - (ξ₁ + ξ₂) * x[2] + ξ₂ * x[4])
            + 3.0 * x[3] - 3.0 * x[1] + 3.0 * ξ₂ * (x[4] - x[2]))/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0)]
end

# mass matrix
B = [] 

# hyperparameters
domainDim = 4
reducedDim = 2
rangeDim = 4
maxOrder = 22
matrixFormat = "full" 
computeFullSpectrum = true
#parametrizationStyle = "normal-form" 
parametrizationStyle = "normal-form-with-resonant-pair"

# initializing and parametrizing system
sys = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            computeFullSpectrum=true,
            parametrizationStyle=parametrizationStyle,
            F=F, B=B, 
            maxOrder=maxOrder);
initialize!(sys);
linearize!(sys);
parametrize!(sys); 

# realify parametrizations
fReal = realify(sys.f, [1,2], invMultiply=true);
WReal = realify(sys.W, [1,2]);

# send results to python and retrieve parametrizations
using PyCall
np = pyimport("numpy");
fCoeffs = np.array(Array(fReal.tensors));
WCoeffs = np.array(Array(WReal.tensors));
np.save("/Users/paolofurlanettoferrari/Documents/Software/Python/globalized-SSM/examples/buckled_beam/fCoeffs.npy", fCoeffs);
np.save("/Users/paolofurlanettoferrari/Documents/Software/Python/globalized-SSM/examples/buckled_beam/WCoeffs.npy", WCoeffs);
f_denom_array = Array(np.load("test/f_denom_array.npy"))
f_numen_array = Array(np.load("test/f_numen_array.npy"))
W_denom_array = Array(np.load("test/W_denom_array.npy"))
W_numen_array = Array(np.load("test/W_numen_array.npy"))

# find Padé approximants of W
N, M = 6,6
WPade = pade_approximant(WReal, N, M);
WPadePython = Tuple{Polynomial, Polynomial}[]
d = Polynomial(WReal.order, WReal.domainDim)
n = Polynomial(WReal.order, WReal.domainDim)
for i in 1:WReal.rangeDim
    d.tensor = SparseArray(W_denom_array[i, :, :])
    n.tensor = SparseArray(W_numen_array[i, :, :])
    push!(WPadePython, (n, d))
end
# check accuracy 
x = rand(ComplexF64, sys.W.domainDim)
evaluate(WReal.tensors[1, :, :], x)
evaluate(WPade[1][1], x)/evaluate(WPade[1][2], x)
evaluate(WPadePython[1][1], x)/evaluate(WPadePython[1][2], x)

# and now for f
N, M = 4,5
fPade = pade_approximant(fReal, N, M);
fPadePython = Tuple{Polynomial, Polynomial}[]
d = Polynomial(fReal.order, fReal.domainDim)
n = Polynomial(fReal.order, fReal.domainDim)
for i in 1:fReal.rangeDim
    d.tensor = SparseArray(f_denom_array[i, :, :])
    n.tensor = SparseArray(f_numen_array[i, :, :])
    push!(fPadePython, (n, d))
end
x = 1 * rand(ComplexF64, sys.W.domainDim)
evaluate(fReal.tensors[1, :, :], x)
evaluate(fPade[1][1], x)/evaluate(fPade[1][2], x)
evaluate(fPadePython[1][1], x)/evaluate(fPadePython[1][2], x)

# compute time responses
using DifferentialEquations
using Plots
include("../test/timeResponse.jl");

t_eval, sol11, sol, W_calc = compare(sys, x0 = [.3 + 0.0im, .3 + 0.0im]);
t_eval_, sol1_, sol2_, W_calc_, sol_pade, W_calc_pade = 
compare(sys, WPades=WPade, fPades=fPade);

# plot time responses
idx = 1
Plots.plot(real(sol[idx, :]), real(sol[idx+1, :]), 
label="Exact", 
linestyle=:dash,
linewidth=2,
color=:black,
guidefont = font(14),
tickfont = font(12),
labelfont = font(12),
#xlims=(-1.5,1.5),
#ylims=(-1.5,1.5),
legendfontsize=10,
xlabel="x₁", 
ylabel="v₁",
framestyle = :box,
legend = :topright)

plot!(real(W_calc_pade[:, idx]), real(W_calc_pade[:, idx+1]), label="k=$maxOrder")

plot!(real(W_calc[:, idx]), real(W_calc[:, idx+1]), label="k=$maxOrder")

cos(maximum(abs.(sol[1,:]-sol[3,:]))im)^2

cos(maximum(abs.(W_calc_pade[:,1]-W_calc_pade[:,3]))im)^2
