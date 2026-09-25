const N_AUX_METRIC_TERMS_COVARIANT_2D = 26

@doc raw"""
    CovariantLinearSystem2D(Nvars, A_matrix_function;
                            global_coordinate_system)

    CovariantLinearSystem2D(Nvars, A_matrix_function, B_matrix_function;
                            global_coordinate_system)

Generic linear system on a 2D manifold embedded in 3D.

Conservative-only form:
```math
\partial_t u + \frac{1}{J}\partial_j\left(J A^j u\right) = 0.
```

Conservative plus nonconservative form:
```math
\partial_t u + \frac{1}{J}\partial_j\left(J A^j u\right) + B^j\partial_j u = 0.
```

`A_matrix_function` and `B_matrix_function` are called during auxiliary-variable
initialization with signatures
`A_matrix_function(x, aux_vars, orientation, equations)` and
`B_matrix_function(x, aux_vars, orientation, equations)`
"""
struct CovariantLinearSystem2D{Nvars,
                               GlobalCoordinateSystem,
                               AMatrixFunction,
                               BMatrixFunction} <:
       TrixiAtmo.AbstractCovariantEquations{2, 3, GlobalCoordinateSystem, Nvars}
    A_matrix_function::AMatrixFunction
    B_matrix_function::BMatrixFunction
    global_coordinate_system::GlobalCoordinateSystem
end

# Conservative constructor
function CovariantLinearSystem2D(Nvars::Integer,
                                 A_matrix_function;
                                 global_coordinate_system = TrixiAtmo.GlobalCartesianCoordinates())
    Nvars_int = Int(Nvars)
    @assert Nvars_int > 0

    return CovariantLinearSystem2D{Nvars_int,
                                   typeof(global_coordinate_system),
                                   typeof(A_matrix_function),
                                   Nothing}(A_matrix_function,
                                            nothing,
                                            global_coordinate_system)
end

# Constructor with nonconservative B matrices
function CovariantLinearSystem2D(Nvars::Integer,
                                 A_matrix_function,
                                 B_matrix_function;
                                 global_coordinate_system = TrixiAtmo.GlobalCartesianCoordinates())
    Nvars_int = Int(Nvars)
    @assert Nvars_int > 0

    return CovariantLinearSystem2D{Nvars_int,
                                   typeof(global_coordinate_system),
                                   typeof(A_matrix_function),
                                   typeof(B_matrix_function)}(A_matrix_function,
                                                              B_matrix_function,
                                                              global_coordinate_system)
end

# -----------------------------------------------------------------------------
# Internal coefficient matrix storage/access
# -----------------------------------------------------------------------------

# Return the first auxiliary-variable index used for A^orientation
@inline function first_aux_index_A(orientation::Integer,
                                   ::CovariantLinearSystem2D{Nvars}) where {Nvars}
    @assert orientation == 1 || orientation == 2
    return N_AUX_METRIC_TERMS_COVARIANT_2D + (orientation - 1) * Nvars * Nvars + 1
end

# Return the first auxiliary-variable index used for B^orientation
@inline function first_aux_index_B(orientation::Integer,
                                   ::CovariantLinearSystem2D{Nvars}) where {Nvars}
    @assert orientation == 1 || orientation == 2
    n_A_entries = 2 * Nvars * Nvars
    return N_AUX_METRIC_TERMS_COVARIANT_2D + n_A_entries +
           (orientation - 1) * Nvars * Nvars + 1
end

# Extract A^orientation from auxiliary variables at one node
@inline function A_matrix(aux_vars,
                          orientation::Integer,
                          equations::CovariantLinearSystem2D{Nvars}) where {Nvars}
    first_index = first_aux_index_A(orientation, equations)
    entries = ntuple(entry -> aux_vars[first_index + entry - 1], Val(Nvars * Nvars))
    return SMatrix{Nvars, Nvars}(entries)
end

# Extract B^orientation from auxiliary variables at one node
@inline function B_matrix(aux_vars,
                          orientation::Integer,
                          equations::CovariantLinearSystem2D{Nvars}) where {Nvars}
    first_index = first_aux_index_B(orientation, equations)
    entries = ntuple(entry -> aux_vars[first_index + entry - 1], Val(Nvars * Nvars))
    return SMatrix{Nvars, Nvars}(entries)
end

# -----------------------------------------------------------------------------
# Equation interface
# -----------------------------------------------------------------------------

function Trixi.varnames(::typeof(Trixi.cons2cons),
                        ::CovariantLinearSystem2D{Nvars}) where {Nvars}
    return ntuple(index -> "u$(index)", Val(Nvars))
end

function Trixi.varnames(::typeof(TrixiAtmo.contravariant2global),
                        ::CovariantLinearSystem2D{Nvars}) where {Nvars}
    return ntuple(index -> "u$(index)", Val(Nvars))
end

@inline function Trixi.cons2entropy(u, aux_vars, equations::CovariantLinearSystem2D)
    return u
end

# Identity coordinate transforms by default
@inline function TrixiAtmo.contravariant2global(u, aux_vars,
                                                equations::CovariantLinearSystem2D)
    return u
end

@inline function TrixiAtmo.global2contravariant(u, aux_vars,
                                                equations::CovariantLinearSystem2D)
    return u
end

# Conservative flux J*A^j*u
@inline function Trixi.flux(u, aux_vars, orientation::Integer,
                            equations::CovariantLinearSystem2D)
    area_element_value = TrixiAtmo.area_element(aux_vars, equations)
    A_local = A_matrix(aux_vars, orientation, equations)
    return area_element_value * (A_local * u)
end

# Nonconservative two-point term for systems with B matrices
# Use a simple left-coefficient/right-state form: J_ll * B_ll * u_rr
@inline function flux_nonconservative(u_ll, u_rr, aux_vars_ll, aux_vars_rr,
                                      orientation::Integer,
                                      equations::CovariantLinearSystem2D)
    J_ll = TrixiAtmo.area_element(aux_vars_ll, equations)
    B_ll = B_matrix(aux_vars_ll, orientation, equations)
    return J_ll * (B_ll * u_rr)
end

# If the B matrix is `Nothing`, then we have no nonconservative flux contribution
@inline function Trixi.have_nonconservative_terms(::CovariantLinearSystem2D{<:Any,
                                                                            <:Any,
                                                                            <:Any,
                                                                            Nothing})
    return Trixi.False()
end

# If the B matrix is not `Nothing`, then we have nonconservative terms
@inline function Trixi.have_nonconservative_terms(::CovariantLinearSystem2D)
    return Trixi.True()
end

# Max wave speed for local Lax-Friedrichs dissipation uses matrix norms as an upper bound
# on eigenvalues of A and B
@inline function max_abs_speed_matrix_contribution(aux_vars_ll, aux_vars_rr,
                                                   orientation::Integer,
                                                   equations::CovariantLinearSystem2D,
                                                   matrix_function)
    matrix_ll = matrix_function(aux_vars_ll, orientation, equations)
    matrix_rr = matrix_function(aux_vars_rr, orientation, equations)
    return max(opnorm(matrix_ll, Inf), opnorm(matrix_rr, Inf))
end

# Version for no nonconservative terms returns zero speed contribution from B matrices
@inline function max_abs_speed_nonconservative_contribution(aux_vars_ll, aux_vars_rr,
                                                            orientation::Integer,
                                                            equations::CovariantLinearSystem2D{<:Any,
                                                                                               <:Any,
                                                                                               <:Any,
                                                                                               Nothing})
    return zero(eltype(aux_vars_ll))
end

# Version for nonconservative terms computes speed contribution from B matrices
@inline function max_abs_speed_nonconservative_contribution(aux_vars_ll, aux_vars_rr,
                                                            orientation::Integer,
                                                            equations::CovariantLinearSystem2D)
    return max_abs_speed_matrix_contribution(aux_vars_ll, aux_vars_rr,
                                             orientation,
                                             equations,
                                             B_matrix)
end

@inline function Trixi.max_abs_speed(u_ll, u_rr, aux_vars_ll, aux_vars_rr,
                                     orientation::Integer,
                                     equations::CovariantLinearSystem2D)
    speed_A = max_abs_speed_matrix_contribution(aux_vars_ll, aux_vars_rr,
                                                orientation,
                                                equations,
                                                A_matrix)
    speed_B = max_abs_speed_nonconservative_contribution(aux_vars_ll, aux_vars_rr,
                                                         orientation, equations)
    return speed_A + speed_B
end

@inline function Trixi.max_abs_speeds(u, aux_vars,
                                      equations::CovariantLinearSystem2D)
    speed_1 = Trixi.max_abs_speed(u, u, aux_vars, aux_vars, 1, equations)
    speed_2 = Trixi.max_abs_speed(u, u, aux_vars, aux_vars, 2, equations)
    return SVector(speed_1, speed_2)
end

# Local Lax-Friedrichs dissipation on all conservative variables
@inline function (dissipation::Trixi.DissipationLocalLaxFriedrichs)(u_ll, u_rr,
                                                                    aux_vars_ll,
                                                                    aux_vars_rr,
                                                                    orientation_or_normal_direction,
                                                                    equations::CovariantLinearSystem2D)
    wave_speed = dissipation.max_abs_speed(u_ll, u_rr, aux_vars_ll, aux_vars_rr,
                                           orientation_or_normal_direction, equations)
    area_element_value = TrixiAtmo.area_element(aux_vars_ll, equations)
    half = convert(eltype(u_ll), 0.5)
    return -half * area_element_value * wave_speed * (u_rr - u_ll)
end

# -----------------------------------------------------------------------------
# Auxiliary-variable count and names
# -----------------------------------------------------------------------------

# Auxiliary variable count includes A/B matrix entries in addition to standard covariant
# geometry variables
@inline function TrixiAtmo.n_aux_node_vars(::CovariantLinearSystem2D{Nvars,
                                                                     <:Any,
                                                                     <:Any,
                                                                     Nothing}) where {Nvars}
    return N_AUX_METRIC_TERMS_COVARIANT_2D + 2 * Nvars * Nvars
end

@inline function TrixiAtmo.n_aux_node_vars(::CovariantLinearSystem2D{Nvars}) where {Nvars}
    # Store A and B for both orientations
    return N_AUX_METRIC_TERMS_COVARIANT_2D + 4 * Nvars * Nvars
end

# Generate auxiliary variable names for A and B matrix entries in the format "A[orientation]
# [row,column]" and "B[orientation][row,column]"
@inline function coefficient_matrix_aux_names(matrix_label::String,
                                              ::CovariantLinearSystem2D{Nvars}) where {Nvars}
    return ntuple(index -> begin
                      orientation = 1 + (index - 1) ÷ (Nvars * Nvars)
                      matrix_linear_index = 1 + (index - 1) % (Nvars * Nvars)
                      row = 1 + (matrix_linear_index - 1) % Nvars
                      column = 1 + (matrix_linear_index - 1) ÷ Nvars
                      "$matrix_label[$orientation][$row,$column]"
                  end,
                  2 * Nvars * Nvars)
end

# Auxiliary variable names include standard covariant geometry variables plus A and 
# optional B matrix entries
@inline function TrixiAtmo.auxvarnames(equations::CovariantLinearSystem2D{<:Any,
                                                                          <:Any,
                                                                          <:Any,
                                                                          Nothing})
    base_names = TrixiAtmo.auxvarnames(TrixiAtmo.CovariantLinearAdvectionEquation2D())
    A_names = coefficient_matrix_aux_names("A", equations)
    return (base_names..., A_names...)
end
@inline function TrixiAtmo.auxvarnames(equations::CovariantLinearSystem2D)
    base_names = TrixiAtmo.auxvarnames(TrixiAtmo.CovariantLinearAdvectionEquation2D())
    A_names = coefficient_matrix_aux_names("A", equations)
    B_names = coefficient_matrix_aux_names("B", equations)
    return (base_names..., A_names..., B_names...)
end

# -----------------------------------------------------------------------------
# Auxiliary-variable initialization
# -----------------------------------------------------------------------------

# No B terms for conservative-only systems
@inline function store_B_matrix_coefficients!(aux_node_vars,
                                              equations::CovariantLinearSystem2D{<:Any,
                                                                                 <:Any,
                                                                                 <:Any,
                                                                                 Nothing},
                                              dg, i, j, element, x_node)
    return nothing
end

# Fill B matrices after A has been stored at this node
@inline function store_B_matrix_coefficients!(aux_node_vars,
                                              equations::CovariantLinearSystem2D{Nvars},
                                              dg, i, j, element, x_node) where {Nvars}
    # Re-read auxiliary variables so B_matrix_function can access stored A values
    aux_node_with_A = TrixiAtmo.get_node_aux_vars(aux_node_vars, equations, dg, i, j,
                                                  element)

    for orientation in 1:2
        B_local = equations.B_matrix_function(x_node, aux_node_with_A, orientation,
                                              equations)
        @assert size(B_local) == (Nvars, Nvars)

        first_index_B = first_aux_index_B(orientation, equations)
        last_index_B = first_index_B + Nvars * Nvars - 1
        aux_node_vars[first_index_B:last_index_B, i, j, element] = SVector(B_local)
    end

    return nothing
end

# Fill A matrices and optional B matrices at every node
function init_auxiliary_node_variables_coefficients!(auxiliary_variables,
                                                     elements,
                                                     equations::CovariantLinearSystem2D{Nvars},
                                                     dg) where {Nvars}
    (; node_coordinates) = elements
    (; aux_node_vars) = auxiliary_variables

    Trixi.@threaded for element in axes(aux_node_vars, 4)
        for j in eachnode(dg), i in eachnode(dg)
            x_node = Trixi.get_node_coords(node_coordinates, equations, dg, i, j, element)
            aux_node = TrixiAtmo.get_node_aux_vars(aux_node_vars, equations, dg, i, j,
                                                   element)

            for orientation in 1:2
                A_local = equations.A_matrix_function(x_node, aux_node, orientation,
                                                      equations)
                @assert size(A_local) == (Nvars, Nvars)

                first_index_A = first_aux_index_A(orientation, equations)
                last_index_A = first_index_A + Nvars * Nvars - 1
                aux_node_vars[first_index_A:last_index_A, i, j, element] = SVector(A_local)
            end

            store_B_matrix_coefficients!(aux_node_vars, equations, dg, i, j, element,
                                         x_node)
        end
    end

    return nothing
end

# Initialize base covariant geometry for arbitrary metric terms
@inline function init_covariant_geometry_aux!(auxiliary_variables,
                                              mesh::P4estMesh{2, 3},
                                              equations::CovariantLinearSystem2D,
                                              dg,
                                              elements,
                                              metric_terms,
                                              bottom_topography)
    invoke(TrixiAtmo.init_auxiliary_node_variables!,
           Tuple{Any,
                 P4estMesh{2, 3},
                 TrixiAtmo.AbstractCovariantEquations{2, 3},
                 Any,
                 Any,
                 Any,
                 Any},
           auxiliary_variables,
           mesh,
           equations,
           dg,
           elements,
           metric_terms,
           bottom_topography)

    return nothing
end

# Initialize base covariant geometry for exact torus metric terms
@inline function init_covariant_geometry_aux!(auxiliary_variables,
                                              mesh::P4estMesh{2, 3},
                                              equations::CovariantLinearSystem2D,
                                              dg,
                                              elements,
                                              metric_terms::MetricTermsCovariantTorus,
                                              bottom_topography)
    invoke(TrixiAtmo.init_auxiliary_node_variables!,
           Tuple{Any,
                 P4estMesh{2, 3},
                 TrixiAtmo.AbstractCovariantEquations{2, 3},
                 Any,
                 Any,
                 MetricTermsCovariantTorus,
                 Any},
           auxiliary_variables,
           mesh,
           equations,
           dg,
           elements,
           metric_terms,
           bottom_topography)

    return nothing
end

@inline function init_covariant_geometry_aux!(auxiliary_variables,
                                              mesh::P4estMesh{2, 3},
                                              equations::CovariantLinearSystem2D,
                                              dg,
                                              elements,
                                              metric_terms::MetricTermsCovariantFace,
                                              bottom_topography)
    invoke(
        TrixiAtmo.init_auxiliary_node_variables!,
        Tuple{
            Any,
            P4estMesh{2, 3},
            TrixiAtmo.AbstractCovariantEquations{2, 3},
            Any,
            Any,
            MetricTermsCovariantFace,
            Any
        },
        auxiliary_variables,
        mesh,
        equations,
        dg,
        elements,
        metric_terms,
        bottom_topography
    )

    return nothing
end

# Initialize geometry first, then store system coefficients A and optional B
@inline function init_auxiliary_node_variables_linear_system!(auxiliary_variables,
                                                              mesh::P4estMesh{2, 3},
                                                              equations::CovariantLinearSystem2D,
                                                              dg,
                                                              elements,
                                                              metric_terms,
                                                              bottom_topography)
    init_covariant_geometry_aux!(auxiliary_variables, mesh, equations, dg, elements,
                                 metric_terms, bottom_topography)

    init_auxiliary_node_variables_coefficients!(auxiliary_variables, elements, equations,
                                                dg)

    return nothing
end

function TrixiAtmo.init_auxiliary_node_variables!(auxiliary_variables,
                                                  mesh::P4estMesh{2, 3},
                                                  equations::CovariantLinearSystem2D,
                                                  dg,
                                                  elements,
                                                  metric_terms,
                                                  bottom_topography)
    return init_auxiliary_node_variables_linear_system!(auxiliary_variables, mesh,
                                                        equations, dg, elements,
                                                        metric_terms, bottom_topography)
end

function TrixiAtmo.init_auxiliary_node_variables!(auxiliary_variables,
                                                  mesh::P4estMesh{2, 3},
                                                  equations::CovariantLinearSystem2D,
                                                  dg,
                                                  elements,
                                                  metric_terms::MetricTermsCovariantTorus,
                                                  bottom_topography)
    return init_auxiliary_node_variables_linear_system!(auxiliary_variables, mesh,
                                                        equations, dg, elements,
                                                        metric_terms, bottom_topography)
end

function TrixiAtmo.init_auxiliary_node_variables!(auxiliary_variables,
                                                  mesh::P4estMesh{2, 3},
                                                  equations::CovariantLinearSystem2D,
                                                  dg,
                                                  elements,
                                                  metric_terms::MetricTermsCovariantFace,
                                                  bottom_topography)
    return init_auxiliary_node_variables_linear_system!(
        auxiliary_variables,
        mesh,
        equations,
        dg,
        elements,
        metric_terms,
        bottom_topography
    )
end