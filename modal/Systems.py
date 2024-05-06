import numpy as np
from dataclasses import dataclass
from typing import Optional
import math
import sympy
from sympy import *
import re


def build_coefficients_matrix() -> np.array:
    return 0

def factorial_sum(last_value: int, n_variables: int, n: int) -> float:
    return sum([(math.factorial(j+n_variables-(n)))/(math.factorial(j)*math.factorial(n_variables-(n))) for j in range(last_value)])

def get_coefficient_column_number(monomial: str, n_variables: int) -> tuple:

    monomial = monomial.replace("**", "^")
    terms = monomial.split("*")
    orders = {}

    for term in terms:
        int_term = int(term[1])
        if int_term not in orders:
            orders[int_term] = 0
        if "^" in term:
            orders[int_term] += int(term[3:])
        else:
            orders[int_term] += 1
    
    order = sum(orders.values())
    orders = dict(sorted(orders.items()))

    powers = []
    for i in range(1,n_variables+1):
        if i in orders:
            powers.append(orders[i])
        else:
            powers.append(0)
    
    last_values = []
    for i in range(n_variables):
        if i == 0:
            last_values.append(order-powers[i])
        else:
            last_values.append(last_values[-1]-powers[i])

    sum_ = 0
    for i in range(len(last_values)):
        s = factorial_sum(last_values[i], n_variables, i+2)
        sum_ = sum_ + s

    return int(order), int(sum_)

def monomial_dict(monomial: str) -> dict:

    monomial = str(monomial)
    monomial = monomial.replace("**", "^")
    terms = monomial.split("*")
    orders = {}

    for term in terms:
        int_term = int(term[1])
        if int_term not in orders:
            orders[int_term] = 0
        if "^" in term:
            orders[int_term] += int(term[3:])
        else:
            orders[int_term] += 1
    
    return orders

def get_order_from_monomial(monomial: str) -> dict:
    
    monomial = str(monomial)
    monomial = monomial.replace("**", "^")
    terms = monomial.split("*")
    orders = {}

    for term in terms:
        int_term = int(term[1])
        if int_term not in orders:
            orders[int_term] = 0
        if "^" in term:
            orders[int_term] += int(term[3:])
        else:
            orders[int_term] += 1
    
    return sum(orders.values())

def get_coefficient_column_from_list(monomial: str, n_variables: int, monomials: list) -> tuple:
    
    for i, m in enumerate(monomials):
        if monomial_dict(monomial) == monomial_dict(m):
            return i
    
    return None


def generate_monomials(variables, order, current_order=0, current_product=1):
    if current_order == order:
        return [current_product]
    else:
        monomials = []
        for v in variables:
            monomials += generate_monomials(variables, order, current_order + 1, current_product * v)
        return monomials

class PolynomialSystem():

    def __init__(self, n_variables: int):
        self.n_variables = n_variables
        self.matrices = {}
        self.LHS = np.ones((n_variables, 1))
        vars = []
        for i in range(n_variables):
            vars.append(symbols(f"x{i}"))
        self.variables = Matrix(vars)
        self.monomials = {}

        print("System created.")
    
    
    def build_coefficients_from_string(self, expression) -> tuple:

        """
        Build a row of the matrices of coefficients corresponding
        to an equation given as a string. 
        
        Inputs:
        ----
        1. String in the form "xn' = <polynomial expression with variables x1, x2, ..., xN>".
        The symbol ' in the LHS means the derivative of the coordinate xn with time.
        2. The total number of variables in the system
        
        Returns:
        ----
        Tuple (x, y) where:
            1. x is the index of the element on the vector corresponding to the derivative in the LHS of the equation
            2. y is a list of tuples (d, R), one for each monomial in the RHS, where d is the order of the monomial 
            and R is a numpy 1D array. Each array is the row for the corresponding matrix of coefficients.
        
        Example:
        ----
        E1, E2 = build_coefficients_from_string("x2' = -x3 + 4.32*x1**2*x3")
        E1 = (1, [0 0 -1])
        E2 = (3, [0 0 4.32 0 0 0 0 0 0 0])
        
        """
        expression = expression.replace(" ", "")
        expression = expression.replace("-x", "-1*x")
        LHS, RHS = expression.split("=")
        raw_monomials = RHS.split("+")

        coefficients = []
        monomials = []
        all_monomials = {}
        for monomial in raw_monomials:
            idx_mult = monomial.find("*")
            coefficients.append(float(monomial[:idx_mult]))
            monomials.append(monomial[idx_mult+1:])

        orders = []
        rows = []
        n_columns = []

        for i, monomial in enumerate(monomials):
            #order, column = get_coefficient_column_number(monomial, n_variables)
            order = get_order_from_monomial(monomial)
            all_monomials[order] = self.make_monomials(order)
            column = get_coefficient_column_from_list(monomial, self.n_variables, all_monomials[order])

            if order not in orders or i == 0:
                last_column = get_coefficient_column_from_list(f"x{self.n_variables-1}^{order}", self.n_variables, all_monomials[order])
                orders.append(order)
                row = np.zeros((last_column + 1))
                row[column] = coefficients[i]
                rows.append(row)
                n_columns.append(last_column + 1)
            else:
                idx = orders.index(order)
                print(idx)
                rows[idx][column] += coefficients[i]
        
        return int(LHS[1]), list(zip(orders, n_columns, rows))

    def add_coefficients_matrix(self, order: int) -> None:
        return None
    
    def add_equation(self, expression) -> None:
        LHS, RHS_items = self.build_coefficients_from_string(expression)
        orders = []
        for order, n_columns, row in RHS_items:
            orders.append(order)
            if order not in self.matrices.keys():
                self.matrices[order] = np.zeros((self.n_variables, n_columns))
                self.matrices[order][LHS-1] = row
            else:
                self.matrices[order][LHS-1] = row
        for order_ in self.matrices.keys():
            if order_ not in orders:
                self.matrices[order_][LHS-1] = np.zeros((self.matrices[order_].shape[1]))
        
        to_delete = []
        for order in self.matrices.keys():
            if not np.any(self.matrices[order]):
                to_delete.append(order)

        for order in to_delete:
            del self.matrices[order]

        self.matrices = dict(sorted(self.matrices.items()))
        self.make_monomials()
        print("Equation added")

        return None
    
    def make_monomials(self, mon_order=None) -> None:
        
        if mon_order == None:
            for order in self.matrices.keys():
                if order == 1: pass
                else:
                    self.monomials[order] = generate_monomials(list(self.variables), order)
        else:
            self.monomials[mon_order] = generate_monomials(list(self.variables), mon_order)
            return self.monomials[mon_order]
        
    
    def diagonalize(self) -> None:

        if self.matrices[1] == {}:
            return None
        
        else:
            new_vars = {}
            for i in range(self.n_variables):
                new_vars[f'x{i}'] = f'u{i}'

            eigvals, eigvecs = np.linalg.eig(self.matrices[1])
            eigvals = np.diag(eigvals)
            eigvec_transform = np.linalg.inv(eigvecs)
            self.diag_matrices = {}
            self.diag_matrices[1] = np.matmul(self.matrices[1], eigvec_transform)
            self.diag_variables = Matrix(eigvec_transform)*self.variables
            self.diag_variables = self.diag_variables.subs(new_vars)         

            diag_monomials = {}
            for order in self.matrices.keys():
                if order == 1: 
                    diag_monomials[order] = expand(Matrix(eigvecs)*(Matrix(self.matrices[order])*self.diag_variables))
                    
                else:
                    dict_subs = {}
                    for i in range(self.n_variables):
                        dict_subs[f"x{i}"] = self.diag_variables[i]
                    
                    diag_monomials[order] = expand(Matrix(eigvecs)*((Matrix(self.matrices[order])*Matrix(self.monomials[order]).subs(dict_subs))))
            
            for n in self.n_variables:
                expression = f"u{i}' = "
                for order in self.matrices.keys():
                    expression += diag_monomials[order][n]
                self.add_equation(expression)

        return None
    
    def print_system(self) -> None:
        orders = self.matrices.keys()
        print_string = "\nX' ="
        for order in orders:
            if order == 1:
                print_string += " LX"
            else:
                print_string += f" + N{order}(X)"
        
        print_string += ", where:"
        print(print_string)

        print("\nX' is the derivative vector with coefficients \n", self.LHS)
        for order in orders:
            if order == 1:
                print("\n\n\n L is a matrix with coefficients \n", self.matrices[1])
            else:
                print(f"\n\n\n N{order}(X) are nonlinear terms with order {order} and matrix coefficients \n", self.matrices[order])


    def normal_form(self) -> None:
        return None


