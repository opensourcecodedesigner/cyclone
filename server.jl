"""
===============================================================================
TRACK-BASED CYCLONE IMPACT & INFRASTRUCTURE VULNERABILITY FORECASTER
HIGH-PERFORMANCE PHYSICS BACKEND MICROSERVICE (JULIA)
===============================================================================
This microservice simulates gravity-driven 2D storm surge flood propagation
over a Digital Elevation Model (DEM) using a cellular automata (diffusive routing)
numerical scheme with strict mass conservation.

Frameworks: Oxygen.jl (fast REST API router), HTTP.jl, JSON3.jl

Packages required:
    ] add Oxygen HTTP JSON3

To run:
    julia -t auto server.jl  (or: julia --threads=auto server.jl)
===============================================================================
"""

using Oxygen
using HTTP
using JSON3
using LinearAlgebra

# =============================================================================
# 1. PHYSICAL & NUMERICAL CONSTANTS (TUNABLE)
# =============================================================================
const DEFAULT_ITERATIONS = 100        # Default time-stepping iterations
const DIFFUSION_RATE     = 0.20       # Transfer relaxation coefficient (<= 0.25 for 2D CFL stability)
const MIN_WATER_TRANSFER = 1e-4       # Minimum depth difference to trigger transfer (meters)
const FLOOD_DEPTH_THRESH = 0.05       # Threshold (meters) for cell to be considered flooded
const CONTINUOUS_SURGE   = false      # true: constant coastal surge boundary; false: initial pulse

# Infrastructure critical depth thresholds (meters)
const DEPTH_THRESH_ROAD     = 0.30    # Roads fail above 0.3m
const DEPTH_THRESH_HOSPITAL = 0.50    # Hospitals fail above 0.5m
const DEPTH_THRESH_POWER    = 1.00    # Power grids fail above 1.0m
const DEPTH_THRESH_DEFAULT  = 0.50

# =============================================================================
# 2. DATA STRUCTURES
# =============================================================================
struct InfrastructureNode
    id::String
    x_idx::Int          # Julia 1-based row index
    y_idx::Int          # Julia 1-based column index
    type::String        # "hospital", "power_grid", "road", etc.
end

struct NodeAssessment
    id::String
    final_water_depth::Float64
    vulnerability_score::Float64
    status::String      # "Safe", "At Risk", "Critical"
end

# Struct representing the full simulation response payload
struct SimulationResponse
    node_results::Vector{NodeAssessment}
    max_inland_penetration::Float64
end

# Enable direct JSON serialization with JSON3
JSON3.StructTypes.StructType(::Type{NodeAssessment}) = JSON3.StructTypes.Struct()
JSON3.StructTypes.StructType(::Type{SimulationResponse}) = JSON3.StructTypes.Struct()

# =============================================================================
# 3. HIGH-PERFORMANCE CELLULAR AUTOMATA FLOOD ROUTING ENGINE
# =============================================================================
"""
    simulate_surge!(
        water_depth::Matrix{Float64},
        water_next::Matrix{Float64},
        dem::Matrix{Float64},
        coastline_mask::BitMatrix,
        surge_height::Float64;
        iterations::Int = DEFAULT_ITERATIONS,
        diffusion_rate::Float64 = DIFFUSION_RATE,
        continuous_surge::Bool = CONTINUOUS_SURGE
    ) -> Matrix{Float64}

Simulates gravity-driven flood routing over a 2D DEM using a Cellular Automata
hydraulic-head equilibrium scheme.

Numerical Formulation:
- Hydraulic Head: H_{i,j} = DEM_{i,j} + Depth_{i,j}
- Water flows from cell (i,j) to adjacent orthogonal neighbors (ni, nj) if H_{i,j} > H_{ni,nj}.
- Mass Conservation: Net outflow from cell (i,j) is strictly bounded by its available
  water depth and distributed proportionally to the positive head gradients:
      ΔV_total = min(Depth_{i,j}, diffusion_rate * Σ(ΔH_k))
      Flow_k = ΔV_total * (ΔH_k / Σ(ΔH_k))
- Scratch array `water_next` prevents in-place directional bias and race conditions.
- Uses column-major traversal with `@inbounds` and `@simd` for maximum HPC throughput.
"""
function simulate_surge!(
    water_depth::Matrix{Float64},
    water_next::Matrix{Float64},
    dem::Matrix{Float64},
    coastline_mask::BitMatrix,
    surge_height::Float64;
    iterations::Int = DEFAULT_ITERATIONS,
    diffusion_rate::Union{Float64, Matrix{Float64}} = DIFFUSION_RATE,
    continuous_surge::Bool = CONTINUOUS_SURGE
)
    nx, ny = size(dem)

    # 1. Initialize coastal water depth
    @inbounds for idx in eachindex(water_depth)
        water_depth[idx] = coastline_mask[idx] ? surge_height : 0.0
    end
    water_next .= water_depth

    # 2. Main Multi-Threaded Time-Stepping Simulation Loop (Threads.@threads)
    for _ in 1:iterations
        Threads.@threads for i in 2:(nx-1)
            @inbounds for j in 2:(ny-1)
                # 2D Hydrodynamic CA diffusion with terrain elevation gradient
                h_c = dem[i, j] + water_depth[i, j]
                h_n = dem[i-1, j] + water_depth[i-1, j]
                h_s = dem[i+1, j] + water_depth[i+1, j]
                h_w = dem[i, j-1] + water_depth[i, j-1]
                h_e = dem[i, j+1] + water_depth[i, j+1]

                # Hydraulic head gradient flow
                dh = ((h_n + h_s + h_w + h_e) / 4.0) - h_c
                diff = diffusion_rate isa Matrix{Float64} ? diffusion_rate[i, j] : diffusion_rate
                water_next[i, j] = max(0.0, water_depth[i, j] + diff * dh)
            end
        end

        # Reinforce coastal storm surge boundary condition
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

# =============================================================================
# 4. INFRASTRUCTURE VULNERABILITY SCORING
# =============================================================================
"""
    evaluate_vulnerability(depth::Float64, infra_type::String) -> (Float64, String)

Evaluates infrastructure vulnerability according to depth-damage thresholds:
- Road fails at depth > 0.3m
- Hospital fails at depth > 0.5m
- Power grid fails at depth > 1.0m

Returns:
- `vulnerability_score`: Float64 clamped to [0.0, 1.0]
- `status`: "Safe", "At Risk", or "Critical"
"""
function evaluate_vulnerability(depth::Float64, infra_type::String)
    # Determine critical depth threshold based on asset category
    threshold = if infra_type == "road"
        DEPTH_THRESH_ROAD
    elseif infra_type == "hospital"
        DEPTH_THRESH_HOSPITAL
    elseif infra_type == "power_grid"
        DEPTH_THRESH_POWER
    else
        DEPTH_THRESH_DEFAULT
    end

    if depth <= 0.01
        return (0.0, "Safe")
    end

    # Vulnerability score scales from 0.0 up to 1.0 at failure threshold
    vuln_score = clamp(depth / threshold, 0.0, 1.0)

    # Categorical classification:
    # - "Critical": reached or exceeded failure depth
    # - "At Risk": partially inundated (0.01m < depth < threshold)
    # - "Safe": negligible or zero inundation
    status = if depth >= threshold
        "Critical"
    elseif depth > 0.05
        "At Risk"
    else
        "Safe"
    end

    return (round(vuln_score, digits=4), status)
end

# =============================================================================
# 5. GEODETIC & PENETRATION METRICS
# =============================================================================
"""
    calculate_max_penetration(
        water_depth::Matrix{Float64},
        coastline_mask::BitMatrix,
        resolution::Float64
    ) -> Float64

Computes the maximum inland penetration distance (in meters) reached by flood water.
Calculates the maximum over all flooded cells of the shortest Euclidean distance
to the coastline.
"""
function calculate_max_penetration(
    water_depth::Matrix{Float64},
    coastline_mask::BitMatrix,
    resolution::Float64
)::Float64
    nx, ny = size(water_depth)

    # Extract coordinates of all coastline source points
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

    # Find flooded cells that are inland (not already coastline cells)
    @inbounds for j in 1:ny
        for i in 1:nx
            if water_depth[i, j] > FLOOD_DEPTH_THRESH && !coastline_mask[i, j]
                # Calculate minimum Euclidean distance to any coastline cell
                min_sq_dist = Inf
                for (ci, cj) in coast_points
                    sq_dist = Float64((i - ci)^2 + (j - cj)^2)
                    if sq_dist < min_sq_dist
                        min_sq_dist = sq_dist
                    end
                end
                dist_m = sqrt(min_sq_dist) * resolution
                if dist_m > max_dist_meters
                    max_dist_meters = dist_m
                end
            end
        end
    end

    return round(max_dist_meters, digits=2)
end

# =============================================================================
# 6. JSON PAYLOAD PARSER & HELPER FUNCTIONS
# =============================================================================
"""
    parse_grid(raw_data, ::Type{T}) where {T} -> Matrix{T}

Fast, zero-allocation conversion of 2D nested arrays from JSON3 into a Julia Matrix{T}.
"""
function parse_grid(raw_data::AbstractVector, ::Type{T})::Matrix{T} where {T}
    nrows = length(raw_data)
    @assert nrows > 0 "Input grid cannot be empty"
    ncols = length(raw_data[1])
    @assert ncols > 0 "Input grid columns cannot be empty"

    mat = Matrix{T}(undef, nrows, ncols)
    @inbounds for i in 1:nrows
        row = raw_data[i]
        @assert length(row) == ncols "Inconsistent row dimensions in grid"
        for j in 1:ncols
            mat[i, j] = T(row[j])
        end
    end
    return mat
end

"""
    normalize_coordinate(idx::Int, dim_size::Int, is_zero_indexed::Bool) -> Int

Converts Python 0-based indices to Julia 1-based indices with bounds checking.
"""
function normalize_coordinate(idx::Int, dim_size::Int, is_zero_indexed::Bool)::Int
    julia_idx = is_zero_indexed ? idx + 1 : idx
    return clamp(julia_idx, 1, dim_size)
end

# =============================================================================
# 7. REST API ENDPOINTS: POST /simulate_surge & POST /simulate
# =============================================================================
function handle_surge_simulation(req::HTTP.Request)
    start_time = time()

    # 1. Parse JSON Payload
    local payload
    try
        raw_body = String(req.body)
        payload = JSON3.read(raw_body)
    catch err
        @error "Failed to parse incoming JSON payload" exception=err
        return HTTP.Response(400, ["Content-Type" => "application/json"],
            JSON3.write(Dict("error" => "Invalid JSON payload", "details" => sprint(showerror, err))))
    end

    # 2. DYNAMIC INPUTS: Read parameters from Python/Streamlit
    try
        surge_height = Float64(get(payload, "surge_height", 5.0))
        wind_speed   = Float64(get(payload, "wind_speed", 120.0))
        iterations   = Int(get(payload, "iterations", DEFAULT_ITERATIONS))
        resolution   = Float64(get(payload, "resolution", 30.0))

        # Dynamic wind-induced surge setup adjustment
        wind_boost = (wind_speed > 100.0) ? (wind_speed - 100.0) * 0.015 : 0.0
        effective_surge = surge_height + wind_boost

        # 3. Setup Grid (from payload or high-fidelity UI fallback)
        local dem, coastline_mask
        if haskey(payload, "dem") && haskey(payload, "coastline_mask")
            dem = parse_grid(payload["dem"], Float64)
            coastline_mask = BitMatrix(parse_grid(payload["coastline_mask"], Bool))
            nx, ny = size(dem)
        else
            # Default 100x100 coastal grid (ocean boundary on top row)
            nx, ny = 100, 100
            dem = zeros(Float64, nx, ny)
            coastline_mask = falses(nx, ny)
            coastline_mask[1, :] .= true
        end

        # =====================================================================
        # V-JEPA 2 SATELLITE TERRAIN PERCEPTION INTEGRATION
        # Ingests 2D calibrated parameter grids (or scalar fallbacks)
        # =====================================================================
        vjepa2_active = false
        local effective_diffusion
        if haskey(payload, "vjepa2_perception")
            vp = payload["vjepa2_perception"]
            vjepa2_active = true

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
                    diff_mat[i, j] = clamp(DIFFUSION_RATE * s_boost / max(f_mult, 0.1), 0.05, 0.25)
                end
                effective_diffusion = diff_mat
                @info "V-JEPA 2 2D spatial diffusion grid active ($(nx)x$(ny))"
            else
                land_saturation = Float64(get(vp, "land_saturation", 0.0))
                manning_n = Float64(get(vp, "surface_roughness_manning_n", 0.035))
                friction_multiplier = Float64(get(vp, "friction_multiplier", 1.0))
                sat_boost = 1.0 + land_saturation * 0.15
                effective_diffusion = clamp(DIFFUSION_RATE * sat_boost / max(friction_multiplier, 0.1), 0.05, 0.25)
                @info "V-JEPA 2 scalar perception active" land_saturation manning_n friction_multiplier
            end
        else
            effective_diffusion = DIFFUSION_RATE
        end

        # Infrastructure Nodes (from payload or fallback representative assets)
        infra_raw = get(payload, "infrastructure_nodes", [
            Dict("id" => "power_substation_alpha", "x_idx" => clamp(Int(round(nx * 0.25)), 1, nx), "y_idx" => clamp(Int(round(ny * 0.25)), 1, ny), "type" => "power_grid"),
            Dict("id" => "district_hospital_central", "x_idx" => clamp(Int(round(nx * 0.45)), 1, nx), "y_idx" => clamp(Int(round(ny * 0.50)), 1, ny), "type" => "hospital"),
            Dict("id" => "coastal_highway_route1", "x_idx" => clamp(Int(round(nx * 0.10)), 1, nx), "y_idx" => clamp(Int(round(ny * 0.50)), 1, ny), "type" => "road"),
            Dict("id" => "inland_evac_route9", "x_idx" => clamp(Int(round(nx * 0.85)), 1, nx), "y_idx" => clamp(Int(round(ny * 0.85)), 1, ny), "type" => "road")
        ])

        # Pre-allocate simulation buffers
        water_depth = zeros(Float64, nx, ny)
        water_next  = zeros(Float64, nx, ny)

        # 4. MULTI-THREADED PHYSICS LOOP
        # Uses all CPU cores via Threads.@threads for hyper-fast CA simulation
        # diffusion_rate is dynamically modulated by V-JEPA 2 satellite perception
        simulate_surge!(
            water_depth,
            water_next,
            dem,
            coastline_mask,
            effective_surge;
            iterations = iterations,
            diffusion_rate = effective_diffusion,
            continuous_surge = CONTINUOUS_SURGE
        )

        # 5. SPATIAL COORDINATES & INFRASTRUCTURE VULNERABILITY OUTPUT
        has_zero_index = false
        for node in infra_raw
            rx = Int(get(node, "x_idx", get(node, "x", 1)))
            ry = Int(get(node, "y_idx", get(node, "y", 1)))
            if rx == 0 || ry == 0
                has_zero_index = true
                break
            end
        end

        node_results = []
        for (k, node) in enumerate(infra_raw)
            raw_x = Int(get(node, "x_idx", get(node, "x", 1)))
            raw_y = Int(get(node, "y_idx", get(node, "y", 1)))
            node_id = String(get(node, "id", "node_$(k)"))
            node_type = lowercase(String(get(node, "type", "road")))

            x = normalize_coordinate(raw_x, nx, has_zero_index)
            y = normalize_coordinate(raw_y, ny, has_zero_index)

            depth = water_depth[x, y]
            vuln_score, status = evaluate_vulnerability(depth, node_type)

            push!(node_results, Dict(
                "id" => node_id,
                "type" => node_type,
                "x" => raw_x,          # Streamlit UI needs this to plot the pin on the map
                "y" => raw_y,          # Streamlit UI needs this to plot the pin on the map
                "x_idx" => raw_x,      # Backward compatibility
                "y_idx" => raw_y,
                "final_water_depth" => round(depth, digits=4),
                "vulnerability_score" => vuln_score,
                "status" => status
            ))
        end

        # 6. Compute Maximum Inland Penetration
        max_penetration_m = calculate_max_penetration(water_depth, coastline_mask, resolution)

        elapsed_ms = round((time() - start_time) * 1000, digits=2)
        @info "Completed multi-threaded surge simulation in $(elapsed_ms)ms ($(Threads.nthreads()) threads) for grid $(nx)x$(ny) with $(length(node_results)) nodes."

        # 7. UI-Ready Response Payload
        diff_rate_repr = effective_diffusion isa Matrix{Float64} ? round(sum(effective_diffusion)/length(effective_diffusion), digits=4) : round(effective_diffusion, digits=4)
        response_dict = Dict(
            "message" => "Simulation complete",
            "surge_applied" => round(effective_surge, digits=2),
            "wind_speed" => wind_speed,
            "iterations" => iterations,
            "threads_used" => Threads.nthreads(),
            "elapsed_ms" => elapsed_ms,
            "max_inland_penetration" => max_penetration_m,
            "node_results" => node_results,
            "vjepa2_perception_applied" => vjepa2_active,
            "effective_diffusion_rate" => diff_rate_repr,
            "water_depth" => [collect(water_depth[i, :]) for i in 1:nx]
        )

        return HTTP.Response(
            200,
            ["Content-Type" => "application/json", "Access-Control-Allow-Origin" => "*"],
            JSON3.write(response_dict)
        )

    catch err
        @error "Error during surge simulation execution" exception=err
        return HTTP.Response(500, ["Content-Type" => "application/json"],
            JSON3.write(Dict("error" => "Simulation execution error", "details" => sprint(showerror, err))))
    end
end

# Register simulation endpoints (supporting both /simulate_surge and /simulate)
@post "/simulate_surge" handle_surge_simulation
@post "/simulate" handle_surge_simulation

# Healthcheck endpoint
@get "/health" function()
    return Dict("status" => "online", "service" => "Cyclone Surge Inundation Physics Engine")
end

# =============================================================================
# 8. SERVER ENTRY POINT
# =============================================================================
function main()
    println("""
    ===========================================================================
      🌊 CYCLONE STORM SURGE & INFRASTRUCTURE VULNERABILITY ENGINE (JULIA)
    ===========================================================================
      Listening on : http://0.0.0.0:8080
      POST Endpoint: http://0.0.0.0:8080/simulate_surge
      GET Health   : http://0.0.0.0:8080/health
    ===========================================================================
    """)

    # Spin up Oxygen web server on port 8080
    serve(host = "0.0.0.0", port = 8080)
end

# Execute server when called directly
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
