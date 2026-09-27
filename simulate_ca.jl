"""
===============================================================================
AEGIS: MULTI-THREADED 2D CELLULAR AUTOMATA HYDRODYNAMICS SOLVER (JULIA)
===============================================================================
High-performance 2D diffusive wave simulation of coastal storm surge.
Accepts input payload via JSON file or stdin:
- surge_height: Peak surge in meters
- iterations: Number of CA diffusion time-steps
- resolution: Grid cell spatial resolution (e.g. 30.0 meters)
- dem: (Optional) 2D elevation grid (defaults to 100x100 flat littoral plain)
- coastline_mask: (Optional) 2D boolean mask of surge boundary (defaults to row 1)
- vjepa2_perception: (Optional)
    - saturation_grid: 2D matrix of soil saturation in [0.0, 1.0]
    - manning_grid: 2D matrix of Manning's roughness in [0.01, 0.15]
    - (Or scalar land_saturation / friction_multiplier)

Outputs JSON:
- water_depth: 2D array of simulated water depths (meters)
- max_inland_penetration: Max flood distance from coast (meters)
- elapsed_ms: Computation latency (ms)
- threads_used: CPU threads utilized
===============================================================================
"""

using JSON3
using LinearAlgebra

const DEFAULT_ITERATIONS = 120
const BASE_DIFFUSION_RATE = 0.20
const FLOOD_DEPTH_THRESH = 0.05
const CONTINUOUS_SURGE = false

function parse_grid(raw_data::AbstractVector, ::Type{T})::Matrix{T} where {T}
    nrows = length(raw_data)
    @assert nrows > 0 "Input grid cannot be empty"
    ncols = length(raw_data[1])
    @assert ncols > 0 "Input grid columns cannot be empty"

    mat = Matrix{T}(undef, nrows, ncols)
    @inbounds for i in 1:nrows
        row = raw_data[i]
        for j in 1:ncols
            mat[i, j] = T(row[j])
        end
    end
    return mat
end

function calculate_max_penetration(water_depth::Matrix{Float64}, coastline_mask::BitMatrix, resolution::Float64)::Float64
    nx, ny = size(water_depth)
    coast_points = Tuple{Int, Int}[]
    @inbounds for j in 1:ny, i in 1:nx
        if coastline_mask[i, j]
            push!(coast_points, (i, j))
        end
    end
    if isempty(coast_points)
        return 0.0
    end

    max_dist_meters = 0.0
    @inbounds for j in 1:ny, i in 1:nx
        if water_depth[i, j] > FLOOD_DEPTH_THRESH && !coastline_mask[i, j]
            min_sq = Inf
            for (ci, cj) in coast_points
                sq = Float64((i - ci)^2 + (j - cj)^2)
                if sq < min_sq
                    min_sq = sq
                end
            end
            dist_m = sqrt(min_sq) * resolution
            if dist_m > max_dist_meters
                max_dist_meters = dist_m
            end
        end
    end
    return round(max_dist_meters, digits=2)
end

function simulate_surge!(
    water_depth::Matrix{Float64},
    water_next::Matrix{Float64},
    dem::Matrix{Float64},
    coastline_mask::BitMatrix,
    surge_height::Float64;
    iterations::Int = DEFAULT_ITERATIONS,
    diffusion_rate::Union{Float64, Matrix{Float64}} = BASE_DIFFUSION_RATE,
    continuous_surge::Bool = CONTINUOUS_SURGE
)
    nx, ny = size(dem)

    # 1. Initialize coastal water depth
    @inbounds for idx in eachindex(water_depth)
        water_depth[idx] = coastline_mask[idx] ? surge_height : 0.0
    end
    water_next .= water_depth

    # 2. Main multi-threaded CA diffusion loop
    for _ in 1:iterations
        Threads.@threads for i in 2:(nx-1)
            @inbounds for j in 2:(ny-1)
                h_c = dem[i, j] + water_depth[i, j]
                h_n = dem[i-1, j] + water_depth[i-1, j]
                h_s = dem[i+1, j] + water_depth[i+1, j]
                h_w = dem[i, j-1] + water_depth[i, j-1]
                h_e = dem[i, j+1] + water_depth[i, j+1]

                dh = ((h_n + h_s + h_w + h_e) / 4.0) - h_c
                diff = diffusion_rate isa Matrix{Float64} ? diffusion_rate[i, j] : diffusion_rate
                water_next[i, j] = max(0.0, water_depth[i, j] + diff * dh)
            end
        end

        if continuous_surge
            @inbounds for idx in eachindex(water_next)
                if coastline_mask[idx]
                    water_next[idx] = max(water_next[idx], surge_height)
                end
            end
        end

        water_depth .= water_next
    end

    return water_depth
end

function run_simulation_from_payload(payload::Dict{String, Any})::Dict{String, Any}
    t0 = time()

    surge_height = Float64(get(payload, "surge_height", 4.2))
    wind_speed   = Float64(get(payload, "wind_speed", 120.0))
    iterations   = Int(get(payload, "iterations", DEFAULT_ITERATIONS))
    resolution   = Float64(get(payload, "resolution", 30.0))

    wind_boost = (wind_speed > 100.0) ? (wind_speed - 100.0) * 0.015 : 0.0
    effective_surge = surge_height + wind_boost

    # Grid setup
    nx, ny = 100, 100
    dem = zeros(Float64, nx, ny)
    coastline_mask = falses(nx, ny)
    coastline_mask[1, :] .= true

    if haskey(payload, "dem") && haskey(payload, "coastline_mask")
        dem = parse_grid(payload["dem"], Float64)
        coastline_mask = BitMatrix(parse_grid(payload["coastline_mask"], Bool))
        nx, ny = size(dem)
    end

    # Diffusion rate calculation (spatially varying vs uniform scalar)
    local diffusion_rate
    vjepa_active = false

    if haskey(payload, "vjepa2_perception")
        vp = payload["vjepa2_perception"]
        vjepa_active = true

        # Check for 2D calibrated grids
        sat_key = haskey(vp, "saturation_grid") ? "saturation_grid" : (haskey(vp, "land_saturation_grid") ? "land_saturation_grid" : nothing)
        man_key = haskey(vp, "manning_grid") ? "manning_grid" : (haskey(vp, "manning_n_grid") ? "manning_n_grid" : (haskey(vp, "surface_roughness_manning_n_grid") ? "surface_roughness_manning_n_grid" : nothing))

        if sat_key !== nothing && man_key !== nothing
            sat_mat = parse_grid(vp[sat_key], Float64)
            man_mat = parse_grid(vp[man_key], Float64)
            diff_mat = zeros(Float64, nx, ny)
            @inbounds for j in 1:ny, i in 1:nx
                s = sat_mat[i, j]
                n = man_mat[i, j]
                f_mult = 1.0 + (n - 0.035) * 12.0
                s_boost = 1.0 + s * 0.15
                diff_mat[i, j] = clamp(BASE_DIFFUSION_RATE * s_boost / max(f_mult, 0.1), 0.05, 0.25)
            end
            diffusion_rate = diff_mat
        else
            # Scalar fallback
            land_sat = Float64(get(vp, "land_saturation", 0.5))
            f_mult   = Float64(get(vp, "friction_multiplier", 1.0))
            sat_boost = 1.0 + land_sat * 0.15
            diffusion_rate = clamp(BASE_DIFFUSION_RATE * sat_boost / max(f_mult, 0.1), 0.05, 0.25)
        end
    else
        diffusion_rate = BASE_DIFFUSION_RATE
    end

    water_depth = zeros(Float64, nx, ny)
    water_next  = zeros(Float64, nx, ny)

    simulate_surge!(
        water_depth,
        water_next,
        dem,
        coastline_mask,
        effective_surge;
        iterations = iterations,
        diffusion_rate = diffusion_rate,
        continuous_surge = Bool(get(payload, "continuous_surge", true))
    )

    max_penetration = calculate_max_penetration(water_depth, coastline_mask, resolution)
    elapsed_ms = round((time() - t0) * 1000, digits=2)

    # Convert 2D matrix to nested vector of vectors for JSON serialization
    depth_grid = [collect(water_depth[i, :]) for i in 1:nx]

    return Dict{String, Any}(
        "message" => "Simulation complete",
        "surge_applied" => round(effective_surge, digits=2),
        "iterations" => iterations,
        "threads_used" => Threads.nthreads(),
        "elapsed_ms" => elapsed_ms,
        "max_inland_penetration" => max_penetration,
        "vjepa2_perception_applied" => vjepa_active,
        "water_depth" => depth_grid
    )
end

function main()
    if length(ARGS) >= 1
        input_file = ARGS[1]
        raw_str = read(input_file, String)
    else
        raw_str = read(stdin, String)
    end

    payload_parsed = JSON3.read(raw_str, Dict{String, Any})
    result = run_simulation_from_payload(payload_parsed)

    out_json = JSON3.write(result)
    if length(ARGS) >= 2
        output_file = ARGS[2]
        write(output_file, out_json)
        println("SUCCESS: Simulation output written to $output_file")
    else
        println(out_json)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
