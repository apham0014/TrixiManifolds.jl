###############################################################################
# Torus wave smoke test in covariant mixed conservative/nonconservative form
###############################################################################
using OrdinaryDiffEq
using Trixi
using TrixiAtmo
using TrixiManifolds
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54

const major_radius = 1.0
const minor_radius = 0.25
const cells_per_major_angle = 8
const cells_per_minor_angle = 4
const final_time = 0.02
const wave_speed_squared = 1.0

@inline function initial_condition_wave(x, t, aux_vars, equations)
    x1, x2, x3 = x
    ring_radius = sqrt(x1^2 + x2^2)
    theta = atan(x2, x1)
    phi = atan(x3, ring_radius - major_radius)

    shape = (1 - 0.25 * cos(phi)) * cos(phi)
    p = 0.1 * shape
    u_theta = 0.0
    u_phi = -0.4 * shape

    sin_theta, cos_theta = sincos(theta)
    sin_phi, cos_phi = sincos(phi)
    basis_theta = SVector(-ring_radius * sin_theta, ring_radius * cos_theta, 0.0)
    basis_phi = SVector(-minor_radius * sin_phi * cos_theta,
                        -minor_radius * sin_phi * sin_theta,
                        minor_radius * cos_phi)

    velocity_global = u_theta * basis_theta + u_phi * basis_phi
    velocity_contravariant = TrixiAtmo.basis_contravariant(aux_vars, equations) *
                             velocity_global

    return SVector(p, velocity_contravariant[1], velocity_contravariant[2])
end

@inline function conservative_coefficient_matrix(x, aux_vars, orientation::Integer,
                                                 equations)
    zero_aux = zero(eltype(aux_vars))
    one_aux = one(eltype(aux_vars))
    if orientation == 1
        return SMatrix{3, 3}(zero_aux, zero_aux, zero_aux,
                             one_aux, zero_aux, zero_aux,
                             zero_aux, zero_aux, zero_aux)
    else
        return SMatrix{3, 3}(zero_aux, zero_aux, zero_aux,
                             zero_aux, zero_aux, zero_aux,
                             one_aux, zero_aux, zero_aux)
    end
end

@inline function nonconservative_coefficient_matrix(x, aux_vars, orientation::Integer,
                                                    equations)
    zero_aux = zero(eltype(aux_vars))
    c2 = convert(eltype(aux_vars), wave_speed_squared)
    Gcon = TrixiAtmo.metric_contravariant(aux_vars, equations)

    if orientation == 1
        b21 = c2 * Gcon[1, 1]
        b31 = c2 * Gcon[2, 1]
    else
        b21 = c2 * Gcon[1, 2]
        b31 = c2 * Gcon[2, 2]
    end

    return SMatrix{3, 3}(zero_aux, b21, b31,
                         zero_aux, zero_aux, zero_aux,
                         zero_aux, zero_aux, zero_aux)
end

equations = CovariantLinearSystem2D(3, conservative_coefficient_matrix,
                                    nonconservative_coefficient_matrix,
                                    global_coordinate_system = GlobalCartesianCoordinates())

volume_flux = (flux_central, flux_nonconservative)
surface_flux = (flux_lax_friedrichs, flux_nonconservative)
solver = DGSEM(polydeg = 2, surface_flux = surface_flux,
               volume_integral = VolumeIntegralFluxDifferencing(volume_flux))

mesh = P4estMeshTorus2D(cells_per_major_angle, cells_per_minor_angle,
                        major_radius, minor_radius,
                        polydeg = Trixi.polydeg(solver))

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_wave, solver,
                                    boundary_conditions = boundary_condition_periodic,
                                    metric_terms = MetricTermsCovariantTorus(major_radius,
                                                                             minor_radius,
                                                                             cells_per_major_angle,
                                                                             cells_per_minor_angle))
ode = semidiscretize(semi, (0.0, final_time))

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 2, save_analysis = false,
                                     extra_analysis_errors = (:conservation_error,))
stepsize_callback = StepsizeCallback(cfl = 0.5)
callbacks = CallbackSet(summary_callback, analysis_callback, stepsize_callback)

sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 5.0e-4, save_everystep = false, callback = callbacks)
