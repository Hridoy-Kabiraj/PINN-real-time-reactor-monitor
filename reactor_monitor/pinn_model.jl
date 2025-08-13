using Lux, DiffEqFlux, Optimization, OptimizationOptimJL, Random, Plots
using ComponentArrays, Zygote, LinearAlgebra, Statistics, Printf
using Sobol, OptimizationOptimisers, ForwardDiff
import DifferentialEquations: ODEProblem, solve, Rodas5 
using Interpolations
using JLD2 # Added for saving/loading model parameters

# Physical parameters
β_i = [0.000213, 0.001413, 0.001264, 0.002548, 0.000742, 0.000271] # Delayed neutron fractions
λ = [0.01244, 0.0305, 0.1114, 0.3013, 1.1361, 3.013] # Decay constants (1/s)
β = sum(β_i) # Total delayed neutron fraction
Λ = 8.13e-5 # Neutron generation time (s)
S₀ = 1e7 # Neutron source (n/cm²·s)

# Time span
t_start = 0.0
t_end = 217.0 # PUR-1 startup takes 217 seconds
tspan = (t_start, t_end)
t_data_all = range(t_start, t_end, length=250)

# Initial conditions
n₀ = 7e4 # Initial neutron density (n/cm³·s)
c₀ = [β_i[i] / (λ[i] * Λ) * n₀ for i in 1:6]
u₀ = [n₀; c₀...] # Initial neutron population and precursor concentrations

# Control rod position and reactivity functions
function control_rod_position(t)
    return (11/60) * t # Convert time to control rod position (cm)
end

function ρ_from_position(x)
    # Convert from pcm to absolute reactivity (divide by 1e5)
    return (-0.0032*x^4 + 0.2564*x^3 - 5.8336*x^2 + 54.353*x - 1168.5) / 1e5
end

function ρ(t)
    x = control_rod_position(t)
    return ρ_from_position(x)
end

# Point kinetics equations
function point_kinetics!(du, u, p, t)
    n, c1, c2, c3, c4, c5, c6 = u
    ρ_t = ρ(t)
    
    # Neutron balance equation
    du[1] = S₀ + (ρ_t - β)/Λ * n + λ[1]*c1 + λ[2]*c2 + λ[3]*c3 + λ[4]*c4 + λ[5]*c5 + λ[6]*c6
    
    # Precursor concentration equations (for each of the 6 groups)
    du[2] = β_i[1]/Λ * n - λ[1]*c1
    du[3] = β_i[2]/Λ * n - λ[2]*c2
    du[4] = β_i[3]/Λ * n - λ[3]*c3
    du[5] = β_i[4]/Λ * n - λ[4]*c4
    du[6] = β_i[5]/Λ * n - λ[5]*c5
    du[7] = β_i[6]/Λ * n - λ[6]*c6
end

# Solve the reference solution numerically
prob = ODEProblem(point_kinetics!, u₀, tspan)
sol_ref = solve(prob, Rodas5(), saveat=t_data_all)

t_vals = sol_ref.t
sol_vals = sol_ref.u

# Extract true data
n_true = [u[1] for u in sol_vals]
c_true = [[u[i] for u in sol_vals] for i in 2:7]

println("Reference solution computed successfully!")
println("Neutron density range: $(minimum(n_true)) to $(maximum(n_true))")

Random.seed!(123)

# Enhanced neural network architecture with 2 inputs: time and control rod position
nn_architecture = Chain(
    Dense(2 => 64, tanh),    # Input: [time, control_rod_position]
    Dense(64 => 128, tanh),
    Dense(128 => 128, tanh),
    Dense(128 => 128, tanh),
    Dense(128 => 64, tanh),
    Dense(64 => 7)  # Output: n, c1, c2, c3, c4, c5, c6
)

# Initialize parameters
rng = Random.default_rng()
ps, st = Lux.setup(rng, nn_architecture)
ps = ComponentArray(ps)

# Normalization constants for better training
t_norm = t_end
x_max = control_rod_position(t_end)  # Maximum control rod position
n_scale = maximum(n_true)
c_scales = [maximum(abs.(c)) for c in c_true]

println("Normalization scales:")
println("Time scale: $t_norm")
println("Control rod position scale: $x_max")
println("Neutron scale: $n_scale")
println("Precursor scales: $c_scales")

# Enhanced neural network wrapper with control rod position input
function neural_net_with_control(t, x, params, st)
    # Normalize inputs
    t_normalized = t / t_norm
    x_normalized = x / x_max
    
    input = [t_normalized, x_normalized]
    output, _ = nn_architecture(input, params, st)
    
    # Apply scaling and ensure positivity for physical variables
    n_pred = abs(output[1]) * n_scale
    c_pred = [abs(output[i+1]) * c_scales[i] for i in 1:6]
    
    return vcat(n_pred, c_pred...)
end

# Physics loss function with control input
function physics_loss_with_control(params, t_points)
    total_loss = 0.0
    
    for t in t_points
        # Get control rod position and reactivity
        x = control_rod_position(t)
        ρ_t = ρ_from_position(x)
        
        # Get predictions using control rod position as input
        u_pred = neural_net_with_control(t, x, params, st)
        n, c1, c2, c3, c4, c5, c6 = u_pred
        
        # Compute derivatives with respect to time
        dudt = ForwardDiff.derivative(t -> begin
            x_t = control_rod_position(t)
            neural_net_with_control(t, x_t, params, st)
        end, t)
        
        # Physics equations residuals
        # Neutron balance equation residual
        residual_n = dudt[1] - (S₀ + (ρ_t - β)/Λ * n + λ[1]*c1 + λ[2]*c2 + λ[3]*c3 + λ[4]*c4 + λ[5]*c5 + λ[6]*c6)
        
        # Precursor equations residuals
        residuals_c = [
            dudt[2] - (β_i[1]/Λ * n - λ[1]*c1),
            dudt[3] - (β_i[2]/Λ * n - λ[2]*c2),
            dudt[4] - (β_i[3]/Λ * n - λ[3]*c3),
            dudt[5] - (β_i[4]/Λ * n - λ[4]*c4),
            dudt[6] - (β_i[5]/Λ * n - λ[5]*c5),
            dudt[7] - (β_i[6]/Λ * n - λ[6]*c6)
        ]
        
        # Add squared residuals
        total_loss += residual_n^2
        for residual in residuals_c
            total_loss += residual^2
        end
    end
    
    return total_loss / length(t_points)
end

# Initial condition loss with control input
function initial_condition_loss_with_control(params)
    x₀ = control_rod_position(0.0)
    u_pred_0 = neural_net_with_control(0.0, x₀, params, st)
    loss = 0.0
    
    # Initial neutron density
    loss += (u_pred_0[1] - u₀[1])^2 / n_scale^2
    
    # Initial precursor concentrations
    for i in 1:6
        loss += (u_pred_0[i+1] - u₀[i+1])^2 / c_scales[i]^2
    end
    
    return loss
end

# Data fitting loss with control input
function data_loss_with_control(params, t_data, u_data)
    loss = 0.0
    n_points = length(t_data)
    n_sample = min(n_points, 25)
    
    for i in 1:n_sample
        idx = min(div((i-1) * (n_points-1), n_sample-1) + 1, n_points)
        t = t_data[idx]
        x = control_rod_position(t)
        u_pred = neural_net_with_control(t, x, params, st)
        u_true = u_data[idx]
        
        # Normalized loss
        loss += (u_pred[1] - u_true[1])^2 / n_scale^2
        for j in 1:6
            loss += (u_pred[j+1] - u_true[j+1])^2 / c_scales[j]^2
        end
    end
    
    return loss / n_sample
end

# Combined loss function with control input
function total_loss_with_control(params, p)
    t_physics = p.t_physics
    t_data = p.t_data
    u_data = p.u_data
    
    l_physics = physics_loss_with_control(params, t_physics)
    l_ic = initial_condition_loss_with_control(params)
    l_data = data_loss_with_control(params, t_data, u_data)
    
    return l_physics + 10.0 * l_ic + 0.1 * l_data
end

# Training data preparation
n_physics_points = 50
t_physics = collect(range(t_start, t_end, length=n_physics_points))
p_data = (t_physics=t_physics, t_data=t_vals, u_data=sol_vals)

callback = function (p, l)
    println("Current loss is: $l")
    return false
end

println("Starting enhanced PINN training with control rod position input...")

# Optimization setup
optfun = OptimizationFunction(total_loss_with_control, Optimization.AutoZygote())
optprob = OptimizationProblem(optfun, ps, p_data)

# Training with ADAM
println("Phase 1: Training with ADAM...")
result1 = solve(optprob, OptimizationOptimisers.Adam(0.001), maxiters=25000, callback=callback)

# Fine-tuning with lower learning rate
println("Phase 2: Fine-tuning with ADAM (lower LR)...")
optprob2 = OptimizationProblem(optfun, result1.u, p_data)
result2 = solve(optprob2, OptimizationOptimisers.Adam(0.0001), maxiters=20000, callback=callback)

trained_params = result2.u

println("Training completed!")
println("Final loss: $(result2.objective)")

# --- Save trained parameters and state for real-time inference ---
# Make sure the directory 'trained_model' exists or create it
mkpath("trained_model") 
@save "trained_model/pinn_params.jld2" trained_params st nn_architecture n_scale c_scales t_norm x_max β Λ S₀ λ β_i u₀ ρ_from_position

println("Trained PINN parameters and state saved to trained_model/pinn_params.jld2")

