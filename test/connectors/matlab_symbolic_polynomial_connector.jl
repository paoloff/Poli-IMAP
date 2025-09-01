function tensor_to_matlab_poly(sys)
    tensors = sys.f.tensors
    n_polys, max_a, max_b = size(tensors)
    matlab_code = ""
    for i in 1:2  # for the first two polynomials
        terms = String[]
        for a in 1:max_a
            for b in 1:max_b
                coeff = tensors[i, a, b]
                if abs(coeff) > 1e-12  # skip zero coefficients
                    term = ""
                    # Format coefficient with parentheses
                    if coeff == 1 && (a > 1 || b > 1)
                        term *= ""
                    elseif coeff == -1 && (a > 1 || b > 1)
                        term *= "-"
                    else
                        term *= "($coeff)"
                    end
                    # Add x^a-1 if needed
                    if a > 1
                        term *= "*x"
                        if a > 2
                            term *= "^$(a-1)"
                        end
                    end
                    # Add y^b-1 if needed
                    if b > 1
                        term *= "*y"
                        if b > 2
                            term *= "^$(b-1)"
                        end
                    end
                    push!(terms, term == "" ? "0" : term)
                end
            end
        end
        poly_expr = join(terms, " + ")
        matlab_code *= "p$i = @(x,y) $poly_expr;\n"
    end
    return matlab_code
end

