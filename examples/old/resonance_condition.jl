# example of resonance_condition function

function resonance_condition(exponent, reducedAxis)

    if reducedAxis == 1
        if exponent[1] == exponent[3] || exponent[1] == exponent[2] + 1

            return true

        end

    elseif reducedAxis == 2   
        if exponent[2] == exponent[4] || exponent[2] == exponent[1] + 1

            return true

        end

    else
        return false
    end
end