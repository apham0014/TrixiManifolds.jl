# metrics.jl
#
# Face-/Gmsh-specific metric-term initialization for a curved
# `P4estMesh{2,3}` produced by `P4estMeshGmshSurface`.
#
# The key idea is:
#
#   - `mesh.tree_node_coordinates[:, i, j, tree]` stores the curved
#     geometry interpolation nodes of each parent p4est tree in R^3.
#   - `mesh.nodes` stores the corresponding 1D interpolation points
#     (for the current converter these are either [-1, 1] for Q1
#     or [-1, 0, 1] for Q2).
#   - Every refined p4est quadrant is only a sub-square of its parent tree.
#   - We therefore:
#       1. map local quadrant coordinates (xi1, xi2) to parent-tree
#          coordinates (u, v) in [-1, 1]^2,
#       2. evaluate the parent high-order geometry map there,
#       3. pass that element-local surface map into the generic
#          differential-geometry initializer in `metrics_2d.jl`.
#

"""
    MetricTermsCovariantFace()

Dispatch tag for a curved face surface represented by the geometry already
stored in a `P4estMesh{2,3}`.

The current `P4estMeshGmshSurface` converter stores the physical high-order
geometry nodes directly inside `mesh.tree_node_coordinates`, so this type does
not need to duplicate any geometry data.
"""
struct MetricTermsCovariantFace end


# =============================================================================
# 1D Lagrange interpolation on the parent p4est tree
# =============================================================================

"""
    face_lagrange_basis(x, nodes)

Evaluate all 1D Lagrange basis functions associated with `nodes` at `x`.

For the current `P4estMeshGmshSurface` converter:

- Q1 geometry uses `nodes = [-1, 1]`
- Q2 geometry uses `nodes = [-1, 0, 1]`

`x` may be an automatic-differentiation number, which is important because
`metrics_2d.jl` differentiates the returned surface map.
"""
@inline function face_lagrange_basis(x, nodes)
    L = map(eachindex(nodes)) do i
        Li = one(x)

        for j in eachindex(nodes)
            if i != j
                Li *= (x - nodes[j]) / (nodes[i] - nodes[j])
            end
        end

        Li
    end

    return SVector(L)
end


# =============================================================================
# Parent-tree high-order geometry map
# =============================================================================

"""
    face_tree_surface_map(mesh, tree_id)

Return a callable map

    (u, v) -> X(u, v) in R^3

for one complete parent p4est tree, where `(u,v) ∈ [-1,1]^2`.

The map is the tensor-product Lagrange interpolant defined by
`mesh.nodes` and `mesh.tree_node_coordinates[:, :, :, tree_id]`.
"""
function face_tree_surface_map(mesh::P4estMesh{2, 3}, tree_id::Integer)
    nodes = mesh.nodes
    tree_node_coordinates = mesh.tree_node_coordinates

    nnodes = length(nodes)

    @assert size(tree_node_coordinates, 1) == 3
    @assert size(tree_node_coordinates, 2) == nnodes
    @assert size(tree_node_coordinates, 3) == nnodes
    @assert 1 <= tree_id <= size(tree_node_coordinates, 4)

    return (u, v) -> begin
        Lu = face_lagrange_basis(u, nodes)
        Lv = face_lagrange_basis(v, nodes)

        X11 = SVector(tree_node_coordinates[1, 1, 1, tree_id],
                      tree_node_coordinates[2, 1, 1, tree_id],
                      tree_node_coordinates[3, 1, 1, tree_id])

        X = (Lu[1] * Lv[1]) * X11

        for j in 1:nnodes, i in 1:nnodes
            (i == 1 && j == 1) && continue

            Xij = SVector(tree_node_coordinates[1, i, j, tree_id],
                          tree_node_coordinates[2, i, j, tree_id],
                          tree_node_coordinates[3, i, j, tree_id])

            X += (Lu[i] * Lv[j]) * Xij
        end

        return X
    end
end


# =============================================================================
# p4est quadrant -> parent-tree reference coordinates
# =============================================================================

"""
    calc_face_element_map_parameters(mesh, RealT)

For every current p4est element, determine its parent tree, lower-left
parent-reference coordinates, and refinement scale.

The parent tree uses `[-1,1]^2`. If a quadrant occupies
`[a,a+s] x [b,b+s]` in p4est's normalized `[0,1]^2` tree coordinates, then

    u = -1 + 2a + s * (xi1 + 1)
    v = -1 + 2b + s * (xi2 + 1),

where `(xi1,xi2) ∈ [-1,1]^2` are local quadrant coordinates.
"""
function calc_face_element_map_parameters(mesh::P4estMesh{2, 3},
                                          ::Type{RealT}) where {RealT <: Real}
    nelements = Trixi.ncells(mesh)

    tree_ids = Vector{Int}(undef, nelements)
    u_origin = Vector{RealT}(undef, nelements)
    v_origin = Vector{RealT}(undef, nelements)
    scales = Vector{RealT}(undef, nelements)

    inv_p4est_root_len =
        ldexp(one(RealT), -Trixi.P4EST_MAXLEVEL)

    trees = Trixi.unsafe_wrap_sc(Trixi.p4est_tree_t, mesh.p4est.trees)

    for tree_id in eachindex(trees)
        tree = trees[tree_id]
        quadrants =
            Trixi.unsafe_wrap_sc(Trixi.p4est_quadrant_t, tree.quadrants)
        tree_offset = Int(tree.quadrants_offset)

        for local_quadrant_id in eachindex(quadrants)
            quad = quadrants[local_quadrant_id]
            element = tree_offset + local_quadrant_id

            scale = p4est_quadrant_reference_scale(quad.level, RealT)

            a = convert(RealT, quad.x) * inv_p4est_root_len
            b = convert(RealT, quad.y) * inv_p4est_root_len

            tree_ids[element] = tree_id
            u_origin[element] = -one(RealT) + 2 * a
            v_origin[element] = -one(RealT) + 2 * b
            scales[element] = scale
        end
    end

    return tree_ids, u_origin, v_origin, scales
end



"""
    TrixiAtmo.init_auxiliary_node_variables!(..., metric_terms::MetricTermsCovariantFace, ...)

Initialize covariant auxiliary variables for a curved face surface.

This method only constructs the element-local surface map. The actual
differential-geometry calculations are delegated to
`init_auxiliary_node_variables_from_map!` from `metrics_2d.jl`.
"""
function TrixiAtmo.init_auxiliary_node_variables!(
    auxiliary_variables,
    mesh::P4estMesh{2, 3},
    equations::TrixiAtmo.AbstractCovariantEquations{2, 3},
    dg,
    elements,
    metric_terms::MetricTermsCovariantFace,
    bottom_topography
)
    @assert equations.global_coordinate_system isa
            TrixiAtmo.GlobalCartesianCoordinates

    RealT = eltype(auxiliary_variables.aux_node_vars)
    one_aux = one(RealT)

    tree_ids, u_origin, v_origin, scales =
        calc_face_element_map_parameters(mesh, RealT)

    surface_map_for_element = element -> begin
        tree_id = tree_ids[element]
        u0 = u_origin[element]
        v0 = v_origin[element]
        scale = scales[element]

        tree_surface_map = face_tree_surface_map(mesh, tree_id)

        return (xi1, xi2) -> begin
            u = u0 + scale * (xi1 + one_aux)
            v = v0 + scale * (xi2 + one_aux)

            return tree_surface_map(u, v)
        end
    end

    init_auxiliary_node_variables_from_map!(
        auxiliary_variables,
        elements,
        mesh,
        equations,
        dg,
        surface_map_for_element,
        bottom_topography
    )

    return nothing
end
