include("../test/rho_psi.jl");

using NonlinearSolve, StaticArrays, NaNMath
using NLsolve, CairoMakie

M = 3;
f = pCache.fPoly.tensors;
plus = true;

function G_wrap(x, ΔΩ, plus)

    c10val = c10(f)

    gamma = gamma_i(f, M)

    c1_ii_val = c1_ii(f, ΔΩ, M)

    d1_ii_val = d1_ii(f, ΔΩ, M)

    return G_rho_omega(x, ΔΩ, gamma, f, M, c10val, c1_ii_val, d1_ii_val; plus=plus)

end

function discriminant_wrap(x, ΔΩ)

    c10val = c10(f)

    gamma = gamma_i(f, M)

    c1_ii_val = c1_ii(f, ΔΩ, M)

    d1_ii_val = d1_ii(f, ΔΩ, M)

    return f1(x, c10val, c1_ii_val, d1_ii_val, M) ^ 2 + 
    f2(x, c10val, c1_ii_val, d1_ii_val, M) ^ 2 - a(x, gamma, f, M) ^ 2
end


rhos = zeros(100);
ΔΩs = collect(range(-2, 2, length=100));

u_prev = 0.23
switched = false

for i in eachindex(rhos)

    u0 = (0.9 * u_prev, 1.1 * u_prev);

    if switched == false
    
        G_wrap_wrap(x, ΔΩ) =  G_wrap(x, ΔΩ, false);
        prob = IntervalNonlinearProblem(G_wrap_wrap, u0, ΔΩs[i]);
        solver = solve(prob);

        if solver.u > 1.1
            switched = true
        end

    else
        G_wrap_wrap(x, ΔΩ) =  G_wrap(x, ΔΩ, true);
        prob = IntervalNonlinearProblem(G_wrap_wrap, u0, ΔΩs[i]);
        solver = solve(prob);
    end

    rhos[i] = solver.u;
    u_prev = rhos[i];

end

fig = Figure()
ax = Axis(fig[1, 1], xlabel="ΔΩ", ylabel="ρ")
lines!(ax, ΔΩs, rhos)
fig

u0 = (0.000, 1.1);

prob = IntervalNonlinearProblem(discriminant_wrap, u0, 0);

solver = solve(prob)

using GLMakie
x1 = collect(range(0.0, 1.1, length=100));
x2 = collect(range(-2, 2, length=100));
Z = [G_wrap(x, y, true) for x in x1, y in x2]; 
Z2 = [G_wrap(x, y, false) for x in x1, y in x2]; 


fig = GLMakie.Figure();
ax = GLMakie.Axis3(fig[1, 1], xlabel="x₁", ylabel="x₂", zlabel="G_wrap");
GLMakie.surface!(ax, x1, x2, Z);
GLMakie.surface!(ax, x1, x2, Z2);
 

# Plot the plane Z=0
GLMakie.surface!(ax, x1, x2, zeros(length(x1), length(x2)), color=:gray, transparency=true, alpha=0.5);

fig

