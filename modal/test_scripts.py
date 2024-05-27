from Systems import *


def add_equations_and_print(expressions=False, n_variables=3):
    
    if not expressions:
        expressions = []
        expressions.append("x0' = 1*x1*x2**2 + -10*x1 +5*x0")
        expressions.append("x1' = 1*x1+ -4.3*x0*x1")
        expressions.append("x2' = 2*x1 + -7*x2 + 0.5*x0*x2**2")

    p = PolynomialSystem(n_variables)

    for e in expressions:
        p.add_equation(e)
    

    p.print_system()
    return p
