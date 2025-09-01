using DifferentialEquations
using Plots

"""
# Define the first system of ODEs
fun = function (x, p, t)
    return [evaluate(sys.f.tensors[i, :, :], x) for i in 1:sys.reducedDim] 
end

# Initial conditions and time span
x0 = [.01 + 0.0im, .01]
tspan = (0.0, 1000.0)
t_eval = range(0.0, 1000.0, length=10000)

# Solve the first system
prob1 = ODEProblem(fun, x0, tspan)
sol1 = solve(prob1, Tsit5(), saveat=t_eval)

# Compute W_calc
W_calc = zeros(ComplexF64, length(t_eval), sys.domainDim)
for i in 1:length(t_eval)
    for j in 1:sys.domainDim
        W_calc[i, j] = evaluate(sys.W.tensors[j, :, :], sol1.u[i]); # compute_polynomial(W_r, sol1.u[i], d, N, k)
    end
end

# Define the second system of ODEs
func2 = function (x, p, t)
    return F(x)
end

# Initial conditions for the second system
x0_2 = W_calc[1, :]

# Solve the second system
prob2 = ODEProblem(func2, x0_2, tspan)
sol2 = solve(prob2, Tsit5(), saveat=t_eval)

# Plot the results
idx = 4
plot(t_eval, real(sol2[idx, :]), label="Direct numerical integration", xlabel="Time (a.u.)", ylabel="θ₁ (rad)")
plot!(t_eval, real(W_calc[:, idx]), label="Method 1")"""


function compare(sys; x0 = [.3 + 0.0im, .3 + 0.0im], WPades = [], fPades = [])

    # Define the first system of ODEs
    fun = function (x, p, t)
        return [evaluate(sys.f.tensors[i, :, :], x) for i in 1:sys.reducedDim] 
        #return [evaluate(sys.f.tensors[i, :, :, :, :], x) for i in 1:sys.reducedDim] 
    end
    
    # Initial conditions and time span
    tspan = (0.0, 5000.0)
    t_eval = range(0.0, 5000.0, length=50000)
    
    # Solve the first system
    prob1 = ODEProblem(fun, x0, tspan)
    sol1 = solve(prob1, Tsit5(), saveat=t_eval)

    # Compute W_calc
    W_calc = zeros(ComplexF64, length(t_eval), sys.domainDim)
    
    for i in 1:length(t_eval)
        for j in 1:sys.domainDim
            W_calc[i, j] = evaluate(sys.W.tensors[j, :, :], sol1.u[i]); 
            #W_calc[i, j] = evaluate(sys.W.tensors[j, :, :, :, :], sol1.u[i]); 
        end
    end

    # Define the second system of ODEs
    func2 = function (x, p, t)
        return F(x)
    end

    # Initial conditions for the second system
    x0_2 = W_calc[1, :]

    # Solve the second system
    prob2 = ODEProblem(func2, x0_2, tspan)
    sol2 = solve(prob2, Tsit5(), saveat=t_eval)

    if fPades != []
        fun_pade = function (x, p, t)
            return [evaluate(fPades[i][1], x) for i in 1:sys.reducedDim] ./
            [evaluate(fPades[i][2], x) for i in 1:sys.reducedDim]
            #return [evaluate(sys.f.tensors[i, :, :, :, :], x) for i in 1:sys.reducedDim] 
        end

        prob_pade = ODEProblem(fun_pade, x0, tspan)
        sol_pade = solve(prob_pade, Tsit5(), saveat=t_eval)

        W_calc_pade = zeros(ComplexF64, length(t_eval), sys.domainDim)

        if WPades != []

            for i in 1:length(t_eval)
                for j in 1:sys.domainDim
                    W_calc_pade[i, j] = evaluate(WPades[j][1], sol_pade.u[i]) /
                    evaluate(WPades[j][2], sol_pade.u[i]); 
                end
            end

        else
            for i in 1:length(t_eval)
                for j in 1:sys.domainDim
                    W_calc_pade[i, j] = evaluate(sys.W.tensors[j, :, :], sol_pade.u[i])
                end
            end
        end

    end

    # Plot the results
    # idx = 4
    #plot(t_eval, real(sol2[idx, :]), label="Direct numerical integration", xlabel="Time (a.u.)", ylabel="θ₁ (rad)")
    #plot!(t_eval, real(W_calc[:, idx]), label="Method 1")
    if fPades == []
        return t_eval, sol1, sol2, W_calc
    else
        return t_eval, sol1, sol2, W_calc, sol_pade, W_calc_pade
    end
end
