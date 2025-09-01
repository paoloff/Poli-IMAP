using LinearAlgebra, NaNMath



function c10(f)
    return f[2, 1, 1, 1, 2, 1]
end

function c1_ii(f, ΔΩ, M)

    c1ii = []

    for i in 2:M
        push!(c1ii, f[2, i, i, 1, 2, 1] + ΔΩ * f[2, i, i, 1, 2, 2])
    end

    return c1ii
end

function d1_ii(f, ΔΩ, M)

    d1ii = []

    for i in 2:M
        push!(d1ii, f[2, i - 1, i + 1, 2, 1, 1] + ΔΩ * f[2, i - 1, i + 1, 2, 1, 2])
    end

    return d1ii
end

function gamma_i(f, M)

    gamma = []

    for i in 2:M
        push!(gamma, f[2, i, i + 1, 1, 1, 1])
    end

    return gamma
end

function a(rho, gamma, f, M)

    return real(f[2, 1, 2, 1, 1, 1])* rho + 
    sum([real(gamma[i]) * rho ^ (2 * i + 1) for i in 1:M-1])

end

function b(rho, gamma, f, M)

    return imag(f[2, 1, 2, 1, 1, 1]) + 
    sum([imag(gamma[i]) * rho ^ (2 * i) for i in 1:M-1])

end

function f1(rho, c10, c1_ii, d1_ii, M)

    return real(c10) + sum([(real(c1_ii[i]) + 
    real(d1_ii[i])) * rho ^ (2 * i) for i in 1:M-1])

end

function f2(rho, c10, c1_ii, d1_ii, M)

    return  imag(c10) + sum([(imag(c1_ii[i]) - 
    imag(d1_ii[i])) * rho ^ (2 * i) for i in 1:M-1])

end

function g1(rho, c10, c1_ii, d1_ii, M)

    return imag(c10) + sum([(imag(c1_ii[i]) + 
    imag(d1_ii[i])) * rho ^ (2 * i) for i in 1:M-1])

end

function g2(rho, c10, c1_ii, d1_ii, M)

    return real(c10) + sum([(real(c1_ii[i]) - 
    real(d1_ii[i])) * rho ^ (2 * i) for i in 1:M-1])

end

function rho_psi_dot(x, omega, gamma, f, M, c10, c1_ii, d1_ii)

    rho = x[1]
    psi = x[2]

    return [
        a(rho, gamma, f, M) + f1(rho, c10, c1_ii, d1_ii, M) * cos(psi) + f2(rho, c10, c1_ii, d1_ii, M) * sin(psi),
        b(rho, gamma, f, M) - omega + (1/rho) * (g1(rho, c10, c1_ii, d1_ii, M) * cos(psi) - g2(rho, c10, c1_ii, d1_ii, M) * sin(psi))
    ]

end

function discriminant(x, gamma, f, M, c10, c1_ii, d1_ii)

    return f1(x, c10, c1_ii, d1_ii, M) ^ 2 + f2(x, c10, c1_ii, d1_ii, M) ^ 2 - a(x, gamma, f, M) ^ 2
    
end


function G_rho_omega(x, ΔΩ, gamma, f, M, c10, c1_ii, d1_ii; plus=true)
    
    disc = discriminant(x, gamma, f, M, c10, c1_ii, d1_ii)

    if true
        if plus
            K = (- f2(x, c10, c1_ii, d1_ii, M) + real(sqrt(Complex(disc)))) / 
            (a(x, gamma, f, M) - f1(x, c10, c1_ii, d1_ii, M))
        else
            K = (- f2(x, c10, c1_ii, d1_ii, M) - real(sqrt(Complex(disc)))) / 
            (a(x, gamma, f, M) - f1(x, c10, c1_ii, d1_ii, M))
        end

        return (b(x, gamma, f, M) - (1.5 + ΔΩ)) * x +
            g1(x, c10, c1_ii, d1_ii, M) * (1 - K ^ 2) / (1 + K ^ 2) -
            g2(x, c10, c1_ii, d1_ii, M) * 2 * K / (1 + K ^ 2)
    else
        return 1.0
    end

end