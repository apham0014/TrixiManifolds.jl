"""
    TrixiManifolds

Extensions for solving PDEs on curved manifolds in the Trixi ecosystem.

This package currently provides torus-manifold support via
[`P4estMeshTorus2D`](@ref), [`MetricTermsCovariantTorus`](@ref), and
[`torus2cartesian`](@ref), together with
[`CovariantLinearSystem2D`](@ref) for linear covariant systems, with example elixirs
available in [`examples_dir`](@ref).

Related sphere/cubed-sphere functionality is provided by TrixiAtmo.jl and demonstrated
in this repository's examples.
"""
module TrixiManifolds

using Trixi
using TrixiAtmo
using LinearAlgebra: det, inv, opnorm
using StaticArrays: SVector

export examples_dir
export CovariantLinearSystem2D
export flux_nonconservative
export P4estMeshTorus2D, MetricTermsCovariantTorus, torus2cartesian
export P4estMeshGmshSurface


const examples_dir = joinpath(@__DIR__, "..", "examples")

include("manifolds/manifolds.jl")
include("equations/equations.jl")

end
