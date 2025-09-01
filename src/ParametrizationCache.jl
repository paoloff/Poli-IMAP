Base.@kwdef mutable struct ParametrizationCache

    # data
    WPoly::PolynomialArray = PolynomialArray() 
    DWPoly::PolynomialArray = PolynomialArray()
    fPoly::PolynomialArray = PolynomialArray()
    DWfPoly::PolynomialArray = PolynomialArray()
    BPoly::PolynomialArray = PolynomialArray()
    BDWfPoly::PolynomialArray = PolynomialArray()

    # indicators
    fullParametrizationDone::Bool = false
    autParametrizationDone::Bool = false
    nonAutParametrizationDone::Bool = false
    bifParametrizationDone::Bool = false

end