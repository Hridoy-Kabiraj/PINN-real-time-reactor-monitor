# real_time_monitor.jl

using Lux, ComponentArrays, LinearAlgebra, Statistics
using JLD2 # For loading trained parameters
using LibSerialPort # For serial communication with Arduino
using GLMakie, Observables # For live plotting
using Base.Threads # For concurrent tasks

# --- Configuration ---
const SERIAL_PORT_NAME = "COM5" # <<< IMPORTANT: CHANGE THIS TO YOUR ARDUINO'S SERIAL PORT
const BAUD_RATE = 115200 # Must match the baud rate in arduino_ultrasonic_sensor.ino
const PLOT_HISTORY_LENGTH = 5000 # Number of data points to display on the live graph
const UPDATE_INTERVAL_MS = 50 # Milliseconds to wait between plot updates (adjust for smoothness vs. responsiveness)

# --- Load Pre-trained PINN Model and Parameters ---
# Ensure pinn_model.jl has been run at least once to create this file
println("Loading trained PINN parameters...")
# All necessary physical constants and functions are loaded from the JLD2 file
@load "trained_model/pinn_params.jld2" trained_params st nn_architecture n_scale c_scales t_norm x_max β Λ S₀ λ β_i u₀

function ρ_from_position(x)
    return (-0.0032*x^4 + 0.2564*x^3 - 5.8336*x^2 + 54.353*x - 1168.5) / 1e5
end

# Re-define the neural network wrapper function.
# This function needs to be available in this script's scope for inference.
function neural_net_with_control(t, x, params, st)
    # Normalize inputs (using scales loaded from the trained model)
    t_normalized = t / t_norm
    x_normalized = x / x_max
    
    input = [t_normalized, x_normalized]
    output, _ = nn_architecture(input, params, st)
    
    # Apply scaling and ensure positivity for physical variables
    n_pred = abs(output[1]) * n_scale
    c_pred = [abs(output[i+1]) * c_scales[i] for i in 1:6]
    
    return vcat(n_pred, c_pred...)
end

# --- Observables for Live Plotting ---
# These Observables will hold the data for the plots and trigger updates
# Initialize Observables with EMPTY VECTORS of the correct type [5, 6]
time_obs = Observable(Vector{Float64}()) 
n_obs = Observable(Vector{Float64}())
c_sum_obs = Observable(Vector{Float64}())
rho_obs = Observable(Vector{Float64}())
x_rod_obs = Observable(Vector{Float64}())

# --- Makie Plotting Setup ---
println("Setting up live plots...")
GLMakie.activate!() # Use GLMakie for interactive, GPU-accelerated plots [7]


fig = Figure(size=(1200, 1000))

# Create axes for each plot using direct keyword arguments for Makie.Axis [8]
ax1 = Makie.Axis(fig[1, 1], title="Neutron Density n(t)", xlabel="Time (s)", ylabel="Neutron Density")
ax2 = Makie.Axis(fig[2, 1], title="Total Precursor Density C(t)", xlabel="Time (s)", ylabel="Concentration")
ax3 = Makie.Axis(fig[3, 1], title="Reactivity ρ(t)", xlabel="Time (s)", ylabel="Reactivity (Δk/k)")
ax4 = Makie.Axis(fig[4, 1], title="Control Rod Position x(t)", xlabel="Time (s)", ylabel="Position (cm)")

# Set x-limits to show 217 seconds of time
GLMakie.xlims!(ax1, 0, 250)
GLMakie.xlims!(ax2, 0, 250)
GLMakie.xlims!(ax3, 0, 250)
GLMakie.xlims!(ax4, 0, 250)


# Set specific y-dimensions for each plot
GLMakie.ylims!(ax1, 65000, 400000) # Neutron Density
GLMakie.ylims!(ax2, 5.5e7, 3e8) # Precursor Sum
GLMakie.ylims!(ax3, -0.2, 0.2)   # Reactivity
GLMakie.ylims!(ax4, 0, 40)   # Control Rod Position

# Plot initial empty lines using Observables. Makie will automatically update these lines
# as the content of the Observables changes
lines!(ax1, time_obs, n_obs, color=:blue, linewidth=2, label="PINN Prediction")
lines!(ax2, time_obs, c_sum_obs, color=:green, linewidth=2, label="PINN Prediction")
lines!(ax3, time_obs, rho_obs, color=:red, linewidth=2, label="Calculated Reactivity")
lines!(ax4, time_obs, x_rod_obs, color=:purple, linewidth=2, label="Sensor Reading")


# Adjust y-limits for better visualization (optional, can be dynamic)
# For neutron density, it can go very high, so log scale might be useful
# ax1.yscale = log10 # Uncomment if you want log scale for neutron density
# ylims!(ax1, 1e4, 1e10) # Adjust based on expected range if using log scale
# ylims!(ax2, 0, 1e10) # Example for precursor sum
# ylims!(ax3, -0.002, 0.002) # Example for reactivity
# ylims!(ax4, 0, 15) # Example for control rod position (0-11 cm range is typical)

display(fig) # Display the figure

# --- Serial Data Reading Task ---
# Use a Channel to safely pass data from the serial reader to the main loop [10, 11]
serial_data_channel = Channel{Float64}(10) # Buffer for 10 readings

function read_serial_data(port_name::String, baud::Int, data_channel::Channel{Float64})
    println("Attempting to open serial port: $port_name at $baud baud...")
    try
        LibSerialPort.open(port_name, baud) do sp
            println("Serial port opened successfully. Reading data...")
            # Set a read timeout to prevent indefinite blocking [12, 13]
            LibSerialPort.set_read_timeout(sp, 1.0) # 1 second timeout 

            while isopen(sp)
                try
                    if LibSerialPort.bytesavailable(sp) > 0
                        try
                            line = readline(sp)
                            distance_str = strip(line)
                            # Skip if empty string after stripping
                            if isempty(distance_str)
                                continue
                            end
                            distance_cm = parse(Float64, distance_str)

                            if 0.0 <= distance_cm <= 400.0
                                put!(data_channel, distance_cm)
                            else
                                @warn "Received out-of-range distance: $distance_cm cm. Skipping."
                            end
                        catch e
                            if isa(e, ArgumentError) || isa(e, TypeError)
                                @warn "Failed to parse serial data. Error: $e"
                            else
                                @warn "Non-fatal error: $e. Retrying..."
                                continue
                            end
                        end
                    else
                        sleep(0.001)
                    end

                catch e
                    if isa(e, LibSerialPort.Timeout)
                        # println("Serial read timeout. No data received.") # Can be noisy
                    elseif isa(e, ArgumentError) || isa(e, TypeError) # Corrected OR operator
                        @warn "Failed to parse serial data: '$line'. Error: $e"
                    else
                        @error "Serial communication error: $e"
                        break # Exit loop on critical error
                    end
                end
            end
        end
    catch e
        @error "Could not open serial port $port_name: $e"
        println("Please check if the Arduino is connected and the port name is correct.")
    end
    println("Serial reading task terminated.")
end

# Spawn the serial reading task on a separate thread [10, 11]
serial_task = Threads.@spawn read_serial_data(SERIAL_PORT_NAME, BAUD_RATE, serial_data_channel)

# --- Main Loop for Inference and Plotting ---
println("Starting real-time inference and plotting...")
current_time = 0.0 # Initialize a time counter for the plots
last_update_time = time() # For controlling plot update rate

while true
    # Check if the serial reading task is still running
    if istaskdone(serial_task) &&!isready(serial_data_channel) # Corrected AND operator
        @error "Serial reading task has stopped. Exiting real-time monitor."
        break
    end

    # Try to take data from the channel without blocking indefinitely
    # If no data, continue to next iteration after a short sleep
    distance_cm = try take!(serial_data_channel) catch; nothing end

    if distance_cm!== nothing
        global current_time += (time() - last_update_time) # Increment time based on actual elapsed time
        global last_update_time = time()

        # Perform PINN inference
        # The PINN model was trained with time and control rod position as inputs
        u_pred = neural_net_with_control(current_time, distance_cm, trained_params, st)
        
        n_val = u_pred[1]
        c_sum_val = sum(u_pred[2:7]) # Sum of all precursor concentrations
        rho_val = ρ_from_position(distance_cm) # Calculate reactivity from current position
        println("t = $current_time, x = $distance_cm")
        println("PINN outputs: neutron density = $n_val, precursor sum = $c_sum_val, reactivity = $rho_val")
        # Append new data to Observables. Note: push! modifies the array *inside* the Observable [5, 6]
        push!(time_obs[], current_time) # Access the array inside the Observable with
        push!(n_obs[], n_val)
        push!(c_sum_obs[], c_sum_val)
        push!(rho_obs[], rho_val)
        push!(x_rod_obs[], distance_cm)

        # Limit the history length for live plotting
        if length(time_obs[]) > PLOT_HISTORY_LENGTH # Access the array inside the Observable with
            deleteat!(time_obs[], 1)
            deleteat!(n_obs[], 1)
            deleteat!(c_sum_obs[], 1)
            deleteat!(rho_obs[], 1)
            deleteat!(x_rod_obs[], 1)
        end

        # Notify Makie that Observables have changed to trigger plot updates [9, 5, 6, 16]
        notify(time_obs)
        notify(n_obs)
        notify(c_sum_obs)
        notify(rho_obs)
        notify(x_rod_obs)
    end

    # Control the update rate of the main loop
    sleep(UPDATE_INTERVAL_MS / 10000.0)
end

println("Real-time monitoring stopped.")