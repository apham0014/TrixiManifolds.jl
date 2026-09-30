###############################################################################
# Dirac Face advection test (covariant form)
#
# This elixir constructs a mesh of Dirac's face, runs covariant linear
# advection with a Cartesian Gaussian initial condition, and
# converts output to VTK.
###############################################################################
using OrdinaryDiffEq
using OrdinaryDiffEqLowStorageRK: CarpenterKennedy2N54
using Trixi, TrixiAtmo, Trixi2Vtk
using LinearAlgebra: cross
using TrixiManifolds


# Sphere geometry and mesh resolution
# const sphere_radius = 1.0
# const cells_per_face_dim = 8
const output_directory = normpath(@__DIR__, "output", "advection_meshed_DiracFace")
const rotation_period = 1.0
const num_revolutions = 4
const final_time = num_revolutions * rotation_period
const save_interval = 0.02

# Gaussian scalar bump and solid-body rotation with period T = 1
const angular_speed = 2pi / rotation_period
const rotation_axis_tilt = pi / 4
const gaussian_height = 1.0
const gaussian_width = 5.0
const center_longitude_0 = pi / 2
const center_latitude_0 = 0.0
const rotation_axis = SVector(-cos(rotation_axis_tilt), 0.0, sin(rotation_axis_tilt))

# Initial condition for scalar Gaussian bump in global Cartesian coordinates
@inline function initial_condition_transport(x, t, aux_vars, equations)
    radius = sqrt(x[1]^2 + x[2]^2 + x[3]^2)
    x_center_0 = SVector(radius * cos(center_latitude_0) * cos(center_longitude_0),
                         radius * cos(center_latitude_0) * sin(center_longitude_0),
                         radius * sin(center_latitude_0))

    # Rotate Gaussian center along the exact solid-body trajectory.
    axis_cross_center = cross(rotation_axis, x_center_0)
    x_center = x_center_0 +
               sin(angular_speed * t) * axis_cross_center +
               (1 - cos(angular_speed * t)) * cross(rotation_axis, axis_cross_center)

    dx1 = x[1] - x_center[1]
    dx2 = x[2] - x_center[2]
    dx3 = x[3] - x_center[3]
    h = gaussian_height * exp(-gaussian_width * (dx1^2 + dx2^2 + dx3^2) / radius^2)

    return SVector(h)
end

# Global Cartesian solid-body velocity on the sphere
@inline function velocity_global_cartesian(x)
    return angular_speed * cross(rotation_axis, x)
end

# Scalar advection coefficients A^1 and A^2 in local contravariant coordinates
@inline function advection_coefficient_matrix(x, aux_vars, orientation::Integer, equations)
    velocity_contravariant = TrixiAtmo.basis_contravariant(aux_vars, equations) *
                             velocity_global_cartesian(x)
    return SMatrix{1, 1}(velocity_contravariant[orientation])
end

# Clear stale output files from previous runs to avoid mixing datasets.
if isdir(output_directory)
    rm(output_directory; recursive = true, force = true)
end
mkpath(output_directory)

# Covariant 1x1 linear system for scalar advection in global Cartesian coordinates
equations = CovariantLinearSystem2D(1, advection_coefficient_matrix,
                                    global_coordinate_system = GlobalCartesianCoordinates())

# DGSEM solver
solver = DGSEM(polydeg = 2, surface_flux = flux_lax_friedrichs,
               volume_integral = VolumeIntegralWeakForm())

# Create sphere mesh in 3D
mesh = P4estMeshGmshSurface("examples/DiracFace6.inp")

semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_transport, solver,
                                    boundary_conditions = boundary_condition_periodic, metric_terms = TrixiManifolds.MetricTermsCovariantFace())
ode = semidiscretize(semi, (0.0, final_time))

summary_callback = SummaryCallback()
analysis_callback = AnalysisCallback(semi, interval = 10, save_analysis = true,
                                     extra_analysis_errors = (:conservation_error,))
save_solution = SaveSolutionCallback(dt = save_interval,
                                     solution_variables = contravariant2global,
                                     save_initial_solution = true,
                                     output_directory = output_directory)
stepsize_callback = StepsizeCallback(cfl = 0.7)
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

# Solve the ODE
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 1.0e-3, save_everystep = false, callback = callbacks)

# Convert to VTU for visualization
trixi2vtk(output_directory * "/solution_*.h5", output_directory = output_directory)
