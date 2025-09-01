include("../src/Poli_IMAP.jl");
using DifferentialEquations
using Plots

domainDim = 4
reducedDim = 2
rangeDim = 4
maxOrder = 8
matrixFormat = "full" 
computeFullSpectrum = true
parametrizationStyle = "normal-form-with-resonant-pair"

function critical_load(ξ::Float64) 
    μ_c = 7/2 - sqrt(2)
    return μ_c + (ξ * ξ) / 2 - ((3 - 2 * sqrt(2)) / 2) * ((ξ - (4 + 5 * sqrt(2)) * ξ)^2) / ((ξ + ξ) * (ξ + 6 * ξ))
end

ξ₁ = 0.25
ξ₂ = 0.25
μ = 1.8 * critical_load(ξ₁)

#=
function F(x)
    return [x[2],

            -1.0 * x[4] * x[4] * sin(x[1] - x[3]) - 2.0 * x[1] + x[3] - (ξ₁ + ξ₂) * x[2] + ξ₂ * x[4] + μ * sin(x[1] - x[3]),

            x[4],
            
            x[2] * x[2] * sin(x[1] - x[3]) - x[3] + x[1] - ξ₂ * (x[4] - x[2])]
end

function B(x)

    b = [1.0, 
        0.0, 
        0.0,
        0.0,
        0.0,
        3.0,
        0.0,
        cos(x[1] - x[3]),
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        cos(x[1] - x[3]),
        0.0,
        1.0]

    return [bi + 0.0 * x[1] for bi in b]

end=#

function F(x)
    return [x[2],

            (cos(x[1] - x[3]) * (-1.0 * x[3] + x[1] - ξ₂ * x[4] + ξ₂ * x[2])
            + sin(x[1] - x[3]) * (x[2] * x[2] * cos(x[1] - x[3]) + x[4] * x[4] - μ)
            + 2.0 * x[1] - x[3] + (ξ₁ + ξ₂) * x[2] - ξ₂ * x[4])/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0),

            x[4],
            
            (sin(x[1] - x[3]) * (-1.0 * x[4] * x[4] * cos(x[1] - x[3]) + μ * cos(x[1] - x[3]) - 3.0 * x[2] * x[2])
            + cos(x[1] - x[3]) * (- 2.0 * x[1] + x[3] - (ξ₁ + ξ₂) * x[2] + ξ₂ * x[4])
            + 3.0 * x[3] - 3.0 * x[1] + 3.0 * ξ₂ * (x[4] - x[2]))/
            (cos(x[1] - x[3]) * cos(x[1] - x[3]) - 3.0)]
end

# Constants
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
            - (κ / m) * x[1] * x[1] * x[1],
            
            (k / m) * x[1] + (-cte * k / m) * x[2] + c2 / m * x[3] - ((c1 + c2) / m) * x[4]]
end

B = [] 

# initializing and finding the linearized system
sys = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            computeFullSpectrum=true,
            parametrizationStyle=parametrizationStyle,
            F=F, B=B, 
            maxOrder=8);

initialize!(sys);
linearize!(sys);
parametrize!(sys);  

t_eval, sol11, sol, W_calc = compare(sys);

# initializing and finding the linearized system

sys2 = System(domainDim=domainDim, 
            reducedDim=reducedDim,
            computeFullSpectrum=true,
            parametrizationStyle=parametrizationStyle, 
            F=F, B=B, 
            maxOrder=6);

initialize!(sys2);
linearize!(sys2);
parametrize!(sys2);

t_eval, sol12, sol2, W_calc2 = compare(sys2);

# initializing and finding the linearized system
sys3 = System(domainDim=domainDim, 
            reducedDim=reducedDim,
            computeFullSpectrum=true,
            parametrizationStyle=parametrizationStyle, 
            F=F, B=B, 
            maxOrder=8);

initialize!(sys3);
linearize!(sys3);
parametrize!(sys3);

t_eval, sol13, sol3, W_calc3 = compare(sys3);

sys4 = System(domainDim=domainDim, 
            reducedDim=reducedDim, 
            computeFullSpectrum=true,
            parametrizationStyle=parametrizationStyle,
            F=F, B=B, 
            maxOrder=10);

initialize!(sys4);
linearize!(sys4);
parametrize!(sys4);

t_eval, sol14, sol4, W_calc4 = compare(sys4);

idx = 1

plotly()

Plots.plot(real(sol[idx, :]), real(sol[idx+1, :]), 
label="Exact", 
linestyle=:dash,
linewidth=2,
color=:black,
guidefont = font(14),
tickfont = font(12),
labelfont = font(12),
legendfontsize=10,
xlabel="x₁", 
ylabel="v₁",
framestyle = :box,
legend = :topright)

plot!(real(W_calc[:, idx]), real(W_calc[:, idx+1]),
label="Order 3")

plot!(real(W_calc2[:, idx]), real(W_calc2[:, idx+2]),
label="Order 5")

plot!(real(W_calc3[:, idx]), real(W_calc3[:, idx+2]),
label="Order 7", color=:red)


Plots.plot(t_eval[1:2000], real(sol3[3, 1:2000]), 
label="Exact", 
linestyle=:dot,
guidefont = font(14),
tickfont = font(12),
labelfont = font(12),
color=:black,
legendfontsize=10,
linewidth=2,
xlabel="t", 
ylabel="x₁",
size=(600,400),
legend = :topright,
framestyle=:box)

plot!(t_eval[1:2000], real(W_calc4[1:2000, 3]),
color=:red, 
label="Order 7")

x = Complex.(collect(-4:0.01:4)) .+ (1im)*Complex.(collect(-4:0.01:4))';
y = conj.(x)

W1 = zeros(4, size(x, 1), size(y, 1));
W2 = zeros(4, size(x, 1), size(y, 1));
W3 = zeros(4, size(x, 1), size(y, 1));
W4 = zeros(4, size(x, 1), size(y, 1));


for (i, row) in enumerate(eachrow(x))
    for (j, row2) in enumerate(eachrow(y))
        for k in 1:4
            W1[k, i, j] = real(evaluate(sys.W.tensors[k, :, :], [x[i, j], y[i, j]])); 
            W2[k, i, j] = real(evaluate(sys2.W.tensors[k, :, :], [x[i, j], y[i, j]])); 
            W3[k, i, j] = real(evaluate(sys3.W.tensors[k, :, :], [x[i, j], y[i, j]])); 
            W4[k, i, j] = real(evaluate(sys4.W.tensors[k, :, :], [x[i, j], y[i, j]])); 
        end
    end
end

plotlyjs()  # or gr()

plt = Plots.surface(W3[2, :, :]', W3[3, :, :]', W3[4, :, :]',
    xlabel="x₂", ylabel="v₁", zlabel="v₂",
    colorbar=false, 
    size=(600,600),
    xguidefont=14,
    yguidefont=14,
    tickfont = font(10),
    ylims=(-4,4),
    xlims=(-2.2,2.2),
    zlims=(-6.2,6.2),
    zguidefont=14,
    lw=3,
    alpha=0.9,
    legendfontsize=10,
    framestyle=:box,
    wireframe=true)

display(plt)

curve_x = [real(W_calc3[i, 2]) for i in 1:400];
curve_y = [real(W_calc3[i, 3]) for i in 1:400];
curve_z = [real(W_calc3[i, 4]) for i in 1:400];

curve_x_r = [real(sol3.u[i, 1][2]) for i in 1:400];
curve_y_r = [real(sol3.u[i, 1][3]) for i in 1:400];
curve_z_r = [real(sol3.u[i, 1][4]) for i in 1:400];


plot!(curve_x, curve_y, curve_z, color=:red, linewidth=3, label="k = 7")
plot!(curve_x_r, curve_y_r, curve_z_r, color=:black,
linewidth=5, linestyle=:dashdot, label="Exact")


Plots.savefig(plt, "ssm1.html")


## Stability graph

using Plots
using LaTeXStrings

ξ_vals = 0:0.05:2
μ_vals = critical_load.(ξ_vals);

plot(ξ_vals, μ_vals,
     xlabel = L"\frac{c}{ml^2 \omega_0}",
     ylabel = L"P/ml^2\omega_0",
     legend = false,
     lw = 3,
     guidefont = font(14),
     tickfont = font(12),
     size = (500, 400),
     ylim = (0, 4.01),
     xlim = (0, 2.01),
     framestyle = :box,
)