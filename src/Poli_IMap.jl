#=

Poli_IMAP: [Poli]'s [I]nvariant [Ma]nifold [P]arametrizer

Written by: Paolo F. Ferrari, 2025

Contact: paoloff@usp.br

=#

#######################################################################################################
#######################################################################################################
#######################################################################################################

# modules from public packges
using LinearAlgebra
using SparseArrayKit
using Base: reduce, product
using ProgressMeter

# modules from downloaded packages
include("../ext/computational-graph-tools/src/ReverseAD.jl");
include("../ext/computational-graph-tools/src/CompGraphs.jl");
using .ReverseAD, .CompGraphs;

# internal modules
include("../src/Polynomials.jl");
include("../src/DAGs.jl");
include("../src/ParametrizationSettings.jl");
include("../src/ParametrizationCache.jl");
include("../src/AutoDiff.jl");
include("../src/Systems.jl");
include("../src/parametrizations/autonomous.jl");
include("../src/parametrizations/nonautonomous.jl");
include("../src/parametrizations/bifurcation.jl");

#######################################################################################################
#######################################################################################################
#######################################################################################################
