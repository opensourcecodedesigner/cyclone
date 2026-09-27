# AEGIS: Autonomous Emergency Generation & Intelligence System
### *Dual-Engine, Local-First Hydrodynamic Telemetry, Satellite Perception & Autonomous AI Dispatch Pipeline*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)
[![Julia](https://img.shields.io/badge/Julia-1.10+-9558B2?style=for-the-badge&logo=julia&logoColor=white)](https://julialang.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org/)
[![Oxygen.jl](https://img.shields.io/badge/Oxygen.jl-REST%20API-teal?style=for-the-badge)](https://github.com/ox-ygen/Oxygen.jl)
[![V-JEPA 2](https://img.shields.io/badge/Meta%20AI-V--JEPA%202%20(ViT--L)-0081FB?style=for-the-badge&logo=meta)](https://github.com/facebookresearch/vjepa2)
[![LangGraph](https://img.shields.io/badge/LangGraph-State%20Machine-FF6F00?style=for-the-badge)](https://langchain-ai.github.io/langgraph/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG%20Grounding%20(BGE--small)-purple?style=for-the-badge)](https://www.llamaindex.ai/)
[![Gemini Flash](https://img.shields.io/badge/Google%20GenAI-Gemini%20Flash-4285F4?style=for-the-badge&logo=google)](https://ai.google.dev/)
[![Copernicus EMS](https://img.shields.io/badge/Copernicus%20EMS-EMSR357%20Benchmark-E26B00?style=for-the-badge)](https://emergency.copernicus.eu/mapping/list-of-activations-rapid)
[![Streamlit](https://img.shields.io/badge/Streamlit-Tactical%20Console-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io/)

---

## Executive Summary & Project Vision

During catastrophic tropical cyclones (Category 4+ Super Cyclones), municipal emergency management consistently breaks down at the point of **coordination latency**. When extreme storm surge breaches low-lying littoral zones:
* Traditional human disaster response relies on fractured phone trees, manual gauge verification, and inter-departmental committee consensus.
* This introduces an unacceptable **45 to 90-minute decision latency window**.
* In dynamic surge events, floodwaters propagate inland at rates exceeding **0.5 to 1.2 m/s**, severing evacuation corridors, submerging high-voltage transmission switchyards, and flooding hospital ground wards before the first official alert is ratified.

**AEGIS (Autonomous Emergency Generation & Intelligence System)** eliminates this bottleneck. Designed as a dual-engine, local-first disaster intelligence pipeline, AEGIS directly couples **satellite computer vision perception**, **high-performance physical hydrodynamics**, and **autonomous multi-tier AI orchestration**. 

By executing 2D cellular automata inundation routing at native bare-metal speeds and delegating triage to a dual-phase cognitive pipeline (System 1 Deterministic Triage + System 2 Deep Reasoning), AEGIS converts raw cyclone barometric, track, and satellite telemetry into cryptographically auditable, **Common Alerting Protocol (CAP)**-compliant tactical orders and **smart-contract parametric insurance settlements** in **under 3 seconds**.

```
                   MANUAL HUMAN DISPATCH TIMELINE (45 - 90 MINUTES)
 [ Surge Influx ] ──► [ Gauge Verification ] ──► [ Committee Consensus ] ──► [ Evac Order ] (TOO LATE)
                                                                                       
                       AEGIS AUTONOMOUS PIPELINE ( < 3 SECONDS )
 [ Satellite / Radar ] ──► [ V-JEPA 2 Vision ] ──► [ Julia CA: ~2s ] ──► [ System 1 ] ──► [ CAP Dispatch ]
                                                                                   └──► [ Parametric Payout ]
```

---

## System Architecture

AEGIS decouples heavy physical compute from probabilistic cognitive reasoning through a strictly isolated microservice boundary.

```mermaid
flowchart TD
    subgraph SATELLITE_GEOJSON ["Raw Meteorological Telemetry & Geospatial Inputs"]
        A1["Cyclone Track Vector (GeoJSON / IBTrACS)"]
        A2["Sustained Wind Speed & Central Pressure"]
        A3["Digital Elevation Model (DEM) & Land-Sea Mask"]
        A4["Pre-Storm Satellite Imagery (Optical / Sentinel SAR)"]
        A5["Critical Infrastructure Inventory (GIS Node Array)"]
    end

    subgraph PERCEPTION ["Perception Stage: Meta V-JEPA 2 (PyTorch ViT-L FP16)"]
        V1["V-JEPA 2 Frozen Latent Feature Extraction (~650ms)"]
        V2["Surface Roughness Estimator (Manning's n: 0.035 - 0.045)"]
        V3["Pre-Storm Land Saturation Index (0.0 - 1.0)"]
        V4["Effective Dynamic Friction Multiplier (e.g. 1.044x)"]
    end

    subgraph ENGINE_JULIA ["Physics Backend: Julia Multi-Threaded Engine (Oxygen.jl :8080)"]
        B1["Dynamic Surge Boundary (Wind Setup Drag & Pressure Deficit)"]
        B2["Cellular Automata 2D Diffusive Wave Routing (Hydraulic Head Equilibrium)"]
        B3["Perception-Modulated Iteration Stepping (e.g. 120 -> 125 iterations)"]
        B4["Mass-Conserved Inundation Matrix (ΔH_k Gradient Flux Allocation)"]
        B5["Node Vulnerability Scoring (Roads: 0.3m | Hospitals: 0.5m | Power: 1.0m)"]
    end

    subgraph INSURANCE ["Parametric Insurance Liquidity Settlement Engine"]
        INS1{"Threshold Trigger Evaluator"}
        INS2["Depth >= 1.0m: FULL_PAYOUT_TRIGGER (100% Policy Liquidity)"]
        INS3["Depth >= 0.3m: PARTIAL_PAYOUT_TRIGGER (50% Policy Liquidity)"]
        INS4["Depth < 0.3m: NO_TRIGGER (0% Policy Liquidity / Safe)"]
    end

    subgraph SYSTEM_1 ["System 1: Rapid Triage Router (LangGraph State Machine / Jev Proxy)"]
        C1{"Sub-Millisecond Triage Evaluator"}
        C2["Status: NOMINAL / SAFE --> Early Compute Halt (0 Token Waste)"]
        C3["Status: CRITICAL / AT RISK --> Emergency Flag Asserted"]
    end

    subgraph SYSTEM_2 ["System 2: Cognitive Reasoning Brain (LlamaIndex + Gemini Flash)"]
        D1["LlamaIndex VectorStoreIndex: Municipal SOPs (BGE-small-en-v1.5)"]
        D2["Grounding Context: High-Voltage De-Energization, Vertical Evac, NDRF Vectors"]
        D3["Gemini Flash: Tactical Synthesis Engine (with Dynamic Model & Backoff)"]
        D4["Common Alerting Protocol (CAP) Incident Dispatch Directive"]
    end

    subgraph CONSOLE ["Cartographic Noir Command Dashboard (Streamlit :8501 + Folium)"]
        E1["Tab 1: 🚨 LIVE INCIDENT OPERATIONS (Interactive Sliders, Presets, Real-Time Map, CAP Console)"]
        E2["Tab 2: 📊 MODEL VALIDATION (Dual GeoJSON Overlay: AEGIS 2D CA vs Copernicus EMSR357)"]
        E3["Parametric Settlement Liquidity Badges (Real-Time Settlement Ledger)"]
    end

    A4 --> V1 --> V2 & V3 --> V4
    A1 & A2 & A3 & A5 --> B1
    V4 --> B3
    B1 --> B2 --> B3 --> B4 --> B5
    B5 --> INS1
    INS1 --> INS2 & INS3 & INS4
    B5 -- "HTTP POST :8080/simulate_surge" --> C1
    C1 -- "Safe" --> C2
    C1 -- "Emergency" --> C3
    C3 --> D1
    D1 --> D2 --> D3
    B5 -.-> D3
    INS1 -.-> D3
    D3 --> D4
    D4 --> E1
    INS1 --> E3
    B4 --> E2
```

---

## Core Subsystems Deep Dive

### 1. Perception Stage: Meta V-JEPA 2 (PyTorch)
* **File:** `perception_stage.py`
* **Stack:** PyTorch 2.6+, CUDA 12.8, `facebookresearch/vjepa2` PyTorch Hub.
* **Context Encoder:** Vision Joint Embedding Predictive Architecture Large (`vjepa2_vit_large` / `ViT-L`, 303.9M parameters, frozen `requires_grad=False`, FP16 precision).
* **Pretraining Architecture:** Uses an Exponential Moving Average (EMA) target encoder to guide latent predictive representations of spatio-temporal blocks without pixel-level reconstruction or information maximization loss.
* **Execution Latency:** $\sim 658 \text{ ms}$ per satellite tile on an NVIDIA RTX GPU; VRAM footprint $\sim 2.1 \text{ GB}$.
* **Physical Parameter Derivation:**
  Instead of feeding black-box pixel values into the hydrodynamic simulation, V-JEPA 2 acts as a self-supervised physical feature encoder:
  * **Pre-Storm Land Saturation ($S_{\text{ground}}$):** Quantifies soil water absorption capacity from optical/SAR latent representations.
  * **Surface Roughness ($n_{\text{Manning}}$):** Maps vegetation, coastal built-up density, and mangroves to fluid friction coefficients ($0.035 - 0.045$).
  * **Effective Friction Multiplier:** Dynamically scales the Julia solver's effective iteration count ($N_{\text{iter}} = \text{round}(120 \times \mu_{\text{friction}})$), realistically adjusting inland water accumulation at critical infrastructure points without distorting macro-scale coastline boundaries.

---

### 2. The Physics Backend: Julia Cellular Automata Engine
* **Stack:** Julia 1.10+, `Oxygen.jl` REST framework, `HTTP.jl`, `JSON3.jl`, `LinearAlgebra`.
* **Port:** `8080` (`/simulate_surge`, `/health`).
* **Execution Latency:** $\sim 2 - 5 \text{ ms}$ internal CA kernel compute time; $\sim 18 \text{ ms}$ total HTTP roundtrip over IPv4 loopback.
* **Hydrodynamic Formulation:**
  AEGIS executes a gravity-driven 2D storm surge flood propagation model over high-resolution Digital Elevation Models (DEM) using a **Cellular Automata (CA)** diffusive routing scheme with strict mass conservation:

  $$\text{Hydraulic Head: } H_{i,j} = \text{DEM}_{i,j} + \text{Depth}_{i,j}$$

  Fluid flows between adjacent orthogonal cells only when a positive head gradient exists ($\Delta H_k = H_{i,j} - H_{ni,nj} > 0$):

  $$\Delta V_{\text{total}} = \min\left(\text{Depth}_{i,j},\; \alpha \cdot \sum_{k} \Delta H_k\right), \quad \text{Flow}_k = \Delta V_{\text{total}} \cdot \frac{\Delta H_k}{\sum_{m} \Delta H_m}$$

  where $\alpha \le 0.25$ guarantees Courant-Friedrichs-Lewy (CFL) numerical stability. In-place directional bias and race conditions are eliminated using dual-buffered matrix rotation (`water_depth` and `water_next`) accelerated with `@inbounds` and `@simd` vectorization.

* **Dynamic Atmospheric Coupling:**
  The coastal surge boundary condition ($S_{\text{eff}}$) couples wind shear stress and barometric pressure drops dynamically:

  $$S_{\text{eff}} = S_{\text{base}} + \Delta S_{\text{wind}} = S_{\text{base}} + (V_{\text{km/h}} > 100 ? (V_{\text{km/h}} - 100) \times 0.015 : 0.0)$$

* **Microservice Reliability & Loopback Optimization:**
  * **Direct IPv4 Loopback (`127.0.0.1`):** Completely eliminates the 2.04-second Windows IPv6 (`::1`) DNS fallback penalty incurred by `localhost` requests against IPv4-only Oxygen listeners.
  * **First-Call JIT Absorption (`45.0s Timeout`):** Extends the read timeout from `25.0s` to `45.0s` with dedicated `ReadTimeout` exception trapping, smoothly absorbing Julia's initial multi-threaded JIT compilation pass under `--pkgimages=no`.
  * **Startup Health-Check Guard:** Proactively queries `GET /health` before simulation dispatch. The Streamlit console locks execution and displays an intuitive *“Julia Engine Starting...”* state until the microservice is fully responsive.

---

### 3. Automated Parametric Insurance Liquidity Settlement Engine
Parametric insurance enables instantaneous, automated catastrophe liquidity payouts without requiring weeks of physical loss adjusting. AEGIS evaluates physical depth telemetry directly from Julia's node vulnerability output against statutory parametric trigger policies:

| Asset Category | Failure Threshold | Trigger Status | Policy Settlement | Operational Directive |
| :--- | :--- | :--- | :--- | :--- |
| **Power Substations** | $h \ge 1.00 \text{ m}$ | `FULL_PAYOUT_TRIGGER` | **100% Liquidity** | Instant breaker trip, switchyard de-energization |
| **Emergency Hospitals** | $h \ge 0.50 \text{ m}$ | `FULL_PAYOUT_TRIGGER` | **100% Liquidity** | Vertical ward evacuation, aux generator prep |
| **Arterial Highways** | $h \ge 0.30 \text{ m}$ | `FULL_PAYOUT_TRIGGER` | **100% Liquidity** | Complete vehicular barricade, transit diversion |
| **Secondary Assets** | $0.30\text{m} \le h < 1.0\text{m}$ | `PARTIAL_PAYOUT_TRIGGER` | **50% Liquidity** | Prepositioning pumps, active telemetry monitoring |
| **Uncompromised Nodes** | $h < 0.30 \text{ m}$ | `NO_TRIGGER` | **0% (Safe)** | Corridors verified open for emergency transit |

* **Expanded Puri Coastal Corridor Inventory (16 Critical GIS Nodes):**
  AEGIS geocodes 16 real-world critical infrastructure assets across the Puri, Odisha coastal grid ($19.74^\circ\text{N} - 19.86^\circ\text{N}$, $85.66^\circ\text{E} - 85.87^\circ\text{E}$) spanning 4 distinct coastal exposure tiers:
  * **4 Power Grid Nodes:** Shoreline switchyards (`samuka_beach_electrical_substation`, `balukhand_transformer_yard`), municipal switching stations (`puri_town_33kv_switching_station`), and elevated inland grid hubs (`malatipatpur_grid_substation`).
  * **4 Medical Facilities:** Beachfront urgent care clinics (`swargadwar_emergency_clinic`), coastal village shelters (`red_cross_cyclone_shelter_pentakota`), referral district hospitals (`puri_district_headquarters_hospital`), and high-ground clinics (`gopabandhu_ayurvedic_hospital`).
  * **5 Arterial Corridors:** Oceanfront transit routes (`swargadwar_coastal_boulevard`, `puri_konark_marine_drive_nh316`), central urban spines (`grand_road_bada_danda_corridor`), coastal lagoon feeders (`chilika_inlet_coastal_feeder`), and elevated expressways (`nh316_bhubaneswar_inland_artery`).
  * **3 Community Relief Assets:** Emergency logistics hubs (`mangalahat_food_grain_depot`), reinforced school shelters (`badasankha_multipurpose_cyclone_shelter`), and municipal drinking water plants (`puri_water_treatment_plant_chandanpur`).

---

### 4. System 1 Triage Router (Python / LangGraph)
* **Role:** Sub-millisecond deterministic safety circuit breaker and proxy for the upcoming "Jev" edge model.
* **Mechanism:** Intercepts Julia's physical telemetry output immediately upon completion. Evaluates the multi-node infrastructure health vector against catastrophic failure criteria.
* **Conditional Graph Switch:**
  * If **Nominal / Safe**: Bypasses LLM compute entirely, halting pipeline execution in microseconds ($0$ API tokens spent, zero extraneous API cost).
  * If **Critical / At-Risk**: Asserts emergency flags, identifies compromised sectors (`POWER`, `MEDICAL`, `TRANSPORT`), and initiates contextual retrieval.

---

### 5. System 2 Reasoning Brain: Gemini Flash + LlamaIndex RAG
* **Role:** High-order contextual deduction and standardized tactical order generation.
* **LlamaIndex Vector Store Architecture:**
  * **Embedding Model:** Local `BAAI/bge-small-en-v1.5` (384-dimensional dense vectors via `llama_index.embeddings.huggingface`), operating completely offline with zero OpenAI key dependency.
  * **Knowledge Base:** Vectorizes municipal disaster protocols in `knowledge_base/` (`visakhapatnam_sop.md`, `vddmp_2026.txt`).
  * **Retrieval Dynamics:** Semantic similarity search (`similarity_top_k=2`) directly maps affected critical infrastructure nodes (e.g., `samuka_beach_electrical_substation`, `puri_district_headquarters_hospital`) to exact emergency protocols (220kV transformer cutoffs, Level 3 vertical ICU evacuation, LMO tank securing, 104.4 MHz emergency radio channels).
  * **Session Caching:** Vector store is indexed once into memory and cached via Streamlit `@st.cache_resource`, ensuring sub-millisecond retrieval latency during interactive runs.
* **Resilient AI Dispatch Architecture:**
  * **Dynamic Model Routing:** Uses `gemini-3.1-flash-lite` (with automatic candidate fallback to `gemini-3.5-flash-lite`), dynamically reflecting the active engine across the UI.
  * **3-Tier Exponential Backoff:** Wraps Google GenAI API calls in an automatic retry loop (`1.0s` $\rightarrow$ `2.0s` $\rightarrow$ `4.0s`) targeting transient `503 UNAVAILABLE` or high-demand spikes.
  * **Graceful Degraded Fallback:** If all retries are exhausted, the system automatically falls back to:
    ```text
    [LIVE AI TEMPORARILY UNAVAILABLE — showing parametric trigger data only]
    ```
    preventing raw tracebacks from ever surfacing to operational commanders while keeping all deterministic physics, telemetry, and parametric insurance payouts fully visible.
* **Output Standard:** Generates structured **Common Alerting Protocol (CAP)** tactical directives containing incident headers, threat evaluations, authoritative parametric trigger blocks, prioritized action directives, and NDRF deployment coordinates.

---

### 6. Tactical UI Console: "Cartographic Noir" (Streamlit + Folium)
* **Design Philosophy:** **Cartographic Noir** — an ultra-dark slate palette (`#0E1117`), structured cards (`#161B22`), muted cyan data readouts, and vivid hazard indicators (emerald safe, amber warning, crimson critical).
* **Dual-Tab Interface:**
  1. 🚨 **LIVE INCIDENT OPERATIONS**:
     * **Interactive Hydrodynamic Controls:** Surge $1.0 - 10.0\text{m}$, Wind $80 - 220\text{kts}$, Iterations $50 - 300$, and instant presets (**Cat 3 (3.2m)**, **Fani Cat 4 (4.2m)**).
     * **Real-Time Engine Health Guard:** Live status badge (`🟢 JULIA HPC ENGINE: READY` / `⏳ Julia Engine Starting / Offline`) with auto-disabled execution trigger during cold-start compilation.
     * **Interactive Map Clustering:** Centered on the Puri coastal grid (`19.810°N, 85.815°E`) with Leaflet `MarkerCluster` (`disableClusteringAtZoom: 14`) that smoothly groups dense urban assets while preserving individual color-coded hazard markers, hover tooltips, and detailed modal popups when zoomed in.
     * **Scrollable Parametric Ledger Card:** Integrated monospace liquidity card (`max-height: 220px; overflow-y: auto;`) with live status counts (`● Full`, `● Partial`, `● Safe`).
     * **Scrollable Telemetry Assessment Table:** Comprehensive matrix (`height=420`) displaying Category badges (`⚡ Power Grid`, `🏥 Medical Facility`, `🛣️ Arterial / Evac Route`), grid cell coordinates `(X, Y)`, flood depths, vulnerability scores, and insurance triggers.
  2. 📊 **MODEL VALIDATION (CYCLONE FANI)**:
     * Dual spatial GeoJSON overlay centered on the Puri landfall zone: AEGIS 2D Cellular Automata simulation (cyan) overlaid on Copernicus EMSR357 satellite radar delineation (amber).
     * Live empirical fit scorecards: IoU, Spatial Overlap/Recall, and Precision.
* **Session Persistence:** Integrated `.env` configuration via `python-dotenv` ensures API credentials persist reliably across server restarts without manual shell exports.

---

## Empirical Benchmark: Cyclone Fani vs. Copernicus EMS EMSR357

To validate the physical simulation against real-world catastrophe ground truth, AEGIS executes an end-to-end historical backtest (`backtest_fani.py`) against **Tropical Cyclone Fani** (May 3, 2019 landfall at Puri, Odisha — peak sustained wind $115\text{ kts}$, storm surge $4.2\text{ m}$).

The simulated flood footprint was benchmarked directly against the **Copernicus Emergency Management Service (EMS) Rapid Mapping Activation EMSR357** (derived from TerraSAR-X and COSMO-SkyMed radar constellation passes):

```
===============================================================================
AEGIS BACKTEST ACCURACY VERIFICATION: PRE vs POST V-JEPA 2 INTEGRATION
Cyclone Fani (May 2019) | Benchmark: Copernicus EMS EMSR357
===============================================================================
SPATIAL METRIC COMPARISON
  Intersection over Union (IoU):   60.6%
  Spatial Overlap / Recall:         81.2%
  Precision:                        70.5%
───────────────────────────────────────────────────────────────────────────────
SPATIAL EXTENTS (km²)
  Ground Truth Radar Extent (EMSR357):     60.23 km²
  AEGIS 2D CA Simulated Footprint:         69.41 km²
  Spatial Intersection:                    48.91 km²
───────────────────────────────────────────────────────────────────────────────
V-JEPA 2 INFLUENCE ON INFRASTRUCTURE INUNDATION DEPTHS
  Asset                                    Baseline Depth   V-JEPA 2 Live Depth   Parametric Status
  samuka_beach_electrical_substation        1.8420m          3.3704m              FULL_PAYOUT (100%)
  puri_konark_marine_drive_nh316            0.8650m          0.9380m              FULL_PAYOUT (100%)
  puri_district_headquarters_hospital       0.6210m          0.5408m              FULL_PAYOUT (100%)
  chilika_inlet_coastal_feeder              0.1850m          0.0000m              NO_TRIGGER (Safe)
===============================================================================
```

> [!NOTE]
> **Spatial Extent vs. Depth Dynamics**: V-JEPA 2 satellite feature extraction preserves the macro-scale spatial boundary ($81.2\%$ flood recall) while refining per-asset depth vectors based on pre-storm ground saturation ($73.1\%$) and Manning's roughness ($0.0386$), ensuring accurate parametric payouts.

---

## Project Structure & File Map

```
d:\julia engine\
├── server.jl                           # High-performance multi-threaded Julia CA physics engine (Oxygen.jl :8080)
├── app.py                              # Streamlit Cartographic Noir console with live operations & validation tabs
├── main.py                             # LangGraph orchestrator (System 1 triage + LlamaIndex RAG + Gemini Flash)
├── backtest_fani.py                    # End-to-end historical backtest validation script for Cyclone Fani
├── perception_stage.py                 # Meta V-JEPA 2 (ViT-L FP16) satellite feature extraction pipeline
├── final_lockdown_verify.py            # End-to-end multi-asset consistency & RAG verification test suite
├── final_lockdown_verification_result.json # Verified audit ledger for all four simulated assets
├── verify_fani.py                      # IBTrACS GeoJSON parser & Folium spatial visualizer
├── test_client.py                      # Automated microservice sanity test client for port 8080
├── metrics_comparison.txt              # Citable benchmark scorecard (Pre vs. Post V-JEPA 2 integration)
├── backtest_metrics.json               # Serialized spatial validation metrics (IoU, Recall, Precision)
├── disaster_dispatch_advisory.json     # Sample synthesized CAP emergency alert JSON payload
├── fani_simulated_flood_extent.geojson # Simulated flood polygon layer for GIS/Folium overlay
├── fani_ground_truth_flood_extent.geojson # Copernicus EMSR357 radar delineation ground truth polygon
├── FANI_IBTRACS_TRACK.geojson          # Official NOAA IBTrACS cyclone trajectory geodata
├── knowledge_base/                     # Municipal SOP knowledge base vectorized by LlamaIndex
│   ├── visakhapatnam_sop.md            # VDMA Cyclonic Inundation SOP (220kV cutoff, ICU evacuation, 104.4 MHz)
│   └── vddmp_2026.txt                  # Visakhapatnam District Disaster Management Protocol (VDDMP-2026)
├── perception_cache/                   # Cached V-JEPA 2 embeddings & friction telemetry
├── Project.toml / Manifest.toml        # Julia package environment specifications
├── requirements.txt                    # Python dependencies
└── .env                                # Local environment configuration (GEMINI_API_KEY)
```

---

## Tech Stack & Dependency Matrix

| Layer | Technology | Version / Spec | Purpose |
| :--- | :--- | :--- | :--- |
| **Physics Core** | [Julia](https://julialang.org/) | `1.10+` | Multi-threaded 2D cellular automata hydrodynamic solver |
| **API Server** | [Oxygen.jl](https://github.com/ox-ygen/Oxygen.jl) | `v1.5+` | High-throughput Julia REST microservice on port 8080 |
| **Serialization**| [JSON3.jl](https://github.com/quinnj/JSON3.jl) | `v1.14+` | Zero-allocation struct-to-JSON serialization |
| **Perception** | [Meta V-JEPA 2](https://github.com/facebookresearch/vjepa2) | `ViT-L / FP16` | Satellite terrain feature extraction & friction tuning |
| **Orchestration**| [LangGraph](https://langchain-ai.github.io/langgraph/) | `>=0.0.20` | Cyclical state machine with conditional routing edges |
| **RAG Retrieval**| [LlamaIndex](https://www.llamaindex.ai/) | `>=0.10.0` | In-memory VectorStoreIndex + BAAI/bge-small-en-v1.5 embeddings |
| **System 2 AI** | [Gemini Flash](https://ai.google.dev/) | `google-genai` | Split-second reasoning and CAP dispatch synthesis (gemini-3.1-flash-lite) |
| **UI Dashboard** | [Streamlit](https://streamlit.io/) | `>=1.30.0` | Cartographic Noir incident control console |
| **Mapping Engine**| [Folium](https://python-visualization.github.io/folium/) | `>=0.15.0` | Geospatial GIS layer with Esri satellite integration |
| **Ground Truth** | [Copernicus EMS](https://emergency.copernicus.eu/) | `EMSR357` | Radar satellite ground truth flood delineation benchmark |
| **Geodata** | [IBTrACS GeoJSON](https://www.ncei.noaa.gov/products/international-best-track-archive) | RFC 7946 | Historical Cyclone Fani trajectory and eye vector data |

---

## Local Installation & Quickstart

### Prerequisites
* **Windows 10/11** (or Linux/macOS)
* **Julia 1.10+**: [Download Official Binary](https://julialang.org/downloads/)
* **Python 3.10+**: Configured with virtual environment support
* **Google Gemini API Key**: [Obtain Key from Google AI Studio](https://aistudio.google.com/)
* *(Optional)* **NVIDIA GPU** with CUDA 12+ for accelerated V-JEPA 2 satellite feature extraction

---

### Step 1: Clone Repository & Setup Python Environment

```powershell
# Clone the repository
git clone https://github.com/opensourcecodedesigner/cyclone.git
cd "cyclone"

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install Python requirements
pip install -r requirements.txt
```

---

### Step 2: Configure Environment Variables (.env)

Create or update `.env` in the project root:

```env
# AEGIS Environment Configuration
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

*(The Streamlit dashboard automatically loads `.env` via `python-dotenv`, ensuring keys persist across server restarts).*

---

### Step 3: Launch the Julia Physics Microservice

To ensure high performance and bypass Windows App Control / dynamic library verification blocks on local systems, invoke Julia with disabled precompiled package images:

```powershell
# Install Julia dependencies (First run only)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Launch the multi-threaded physics microservice on port 8080
julia --pkgimages=no --project=. --threads=auto server.jl
```

Once initialized, the service outputs:
```text
===========================================================================
  🌊 CYCLONE STORM SURGE & INFRASTRUCTURE VULNERABILITY ENGINE (JULIA)
===========================================================================
  Listening on : http://0.0.0.0:8080
  POST Endpoint: http://0.0.0.0:8080/simulate_surge
  GET Health   : http://0.0.0.0:8080/health
===========================================================================
```

Verify server health via IPv4 loopback:
```powershell
curl http://127.0.0.1:8080/health
# Returns: {"status":"online","service":"Cyclone Surge Inundation Physics Engine"}
```

---

### Step 4: Launch the Cartographic Noir Tactical Console

Before launching, ensure only a single instance of Streamlit runs to prevent duplicate API calls or port contention:

```powershell
# Verify no duplicate background instances are running
Get-Process python, streamlit -ErrorAction SilentlyContinue | Select-Object Id, ProcessName

# Launch the console (in a terminal with .venv active)
streamlit run app.py
```

Open `http://localhost:8501` in your browser:
* **🚨 LIVE INCIDENT OPERATIONS**:
  * The sidebar dynamically queries `http://127.0.0.1:8080/health`. If Julia is still compiling or launching, it displays `⏳ Julia Engine Starting / Offline` and safely disables the simulation button until the microservice is ready (`🟢 JULIA HPC ENGINE: READY`).
  * Adjust surge and wind sliders or click **Fani Cat 4 (4.2m)** / **Cat 3 (3.2m)** presets.
  * *API Quota Protection*: Moving sliders or switching tabs updates `st.session_state` locally without triggering LLM calls or Julia physics.
  * Click **🚀 EXECUTE LIVE SIMULATION** to trigger the Julia hydrodynamic simulation, evaluate all 16 geocoded Puri infrastructure nodes, inspect the clustered Folium map, and review the resilient Gemini dispatch and scrollable parametric ledger.
* **📊 MODEL VALIDATION**:
  * Inspect the empirical Copernicus radar ground truth overlay (EMSR357) against the 2D Cellular Automata simulation.
  * Dynamically bound to `backtest_metrics.json` displaying verified benchmark figures (**60.6% IoU**, **81.2% Overlap Recall**).

---

### Step 5: (Optional) Run the Cyclone Fani Historical Backtest

To execute the complete end-to-end benchmark from terminal (V-JEPA 2 + Julia + LangGraph + Gemini + Copernicus Evaluation):

```powershell
python backtest_fani.py
```

---

## Production Sample: Common Alerting Protocol (CAP) Output

```markdown
**[INCIDENT HEADER]**
**AEGIS DISASTER COMMAND // SYSTEM ID: VDMA-2026-ALPHA**
**STATUS:** ACTIVE CYCLONE EMERGENCY
**METEOROLOGICAL DATA:** PEAK SURGE 5.0M // SUSTAINED WIND 135 KNOTS
**COMMANDER:** CHIEF AUTONOMOUS INCIDENT COMMANDER (AEGIS)

---

**[CRITICAL THREAT EVALUATION]**
Infrastructure failure imminent. Coastal surge breach has inundated primary littoral sectors across the Puri corridor. **samuka_beach_electrical_substation** is compromised (4.09m depth); **swargadwar_coastal_boulevard** is fully non-traversable (3.82m depth); **puri_konark_marine_drive_nh316** is breached (1.85m depth). **puri_district_headquarters_hospital** is under active flood threat (0.54m depth), requiring immediate vertical ward evacuation. Elevated inland corridors (**nh316_bhubaneswar_inland_artery**) remain intact and confirmed as primary evacuation lifelines.

---

**[PARAMETRIC TRIGGER STATUS]**
• samuka_beach_electrical_substation     : 4.0883m depth | STATUS: FULL_PAYOUT_TRIGGER (100%)
• swargadwar_emergency_clinic            : 2.7410m depth | STATUS: FULL_PAYOUT_TRIGGER (100%)
• swargadwar_coastal_boulevard           : 3.8190m depth | STATUS: FULL_PAYOUT_TRIGGER (100%)
• puri_konark_marine_drive_nh316         : 1.8520m depth | STATUS: FULL_PAYOUT_TRIGGER (100%)
• balukhand_transformer_yard             : 0.6193m depth | STATUS: PARTIAL_PAYOUT_TRIGGER (50%)
• puri_district_headquarters_hospital    : 0.5408m depth | STATUS: FULL_PAYOUT_TRIGGER (100%)
• grand_road_bada_danda_corridor         : 0.4200m depth | STATUS: PARTIAL_PAYOUT_TRIGGER (50%)
• mangalahat_food_grain_depot            : 0.2510m depth | STATUS: NO_TRIGGER (0%)
• badasankha_multipurpose_cyclone_shelter: 0.0400m depth | STATUS: NO_TRIGGER (0%)
• nh316_bhubaneswar_inland_artery        : 0.0000m depth | STATUS: NO_TRIGGER (0%)
• malatipatpur_grid_substation           : 0.0000m depth | STATUS: NO_TRIGGER (0%)

---

**[MANDATORY ACTION DIRECTIVES]**

*   **POWER DIVISION**: 
    *   Execute IMMEDIATE grid de-energization of **samuka_beach_electrical_substation** to prevent flashover. 
    *   Isolate coastal distribution feeders; reroute critical hospital telemetry loads to elevated inland grid at **malatipatpur_grid_substation**. 
    *   Deploy mobile emergency DG generation units immediately.
*   **MEDICAL DIVISION**: 
    *   Initiate emergency vertical patient evacuation at **puri_district_headquarters_hospital** to Floor 2+.
    *   Transition life support to auxiliary battery/rooftop generators; secure oxygen supply systems.
*   **TRANSPORT DIVISION**: 
    *   Enforce absolute vehicular closure along **swargadwar_coastal_boulevard** and **puri_konark_marine_drive_nh316**. 
    *   Funnel all transit and relief logistics through **nh316_bhubaneswar_inland_artery**.
    *   Preposition heavy recovery and water-rescue assets at key bypass junctions.

---

**[NDRF DEPLOYMENT]**
*   **Mission Profile**: High-clearance amphibious transit & rapid triage extraction.
*   **Target**: **puri_district_headquarters_hospital** & **red_cross_cyclone_shelter_pentakota**.
*   **Objective**: Rapid casualty extraction and transport to elevated relief facilities.
*   **Authority**: VDDMP-2026 // Autonomous Incident Override Active.

**END OF DISPATCH // AEGIS COMMAND**
```

---

## Technical Roadmap & Research Horizons

```
[ Active Production Stack ] ──────────────► [ Phase 2: Q4 2026 ] ──────────────► [ Phase 3: 2027 ]
  • Julia 2D Cellular Automata                • Standalone "Jev" Edge Model             • Closed-Loop Autonomous
  • Meta V-JEPA 2 Satellite Vision              (Sub-10M quantized local model)           SCADA / Substation Trip
  • LangGraph State Machine                   • 3D Shallow Water Navier-Stokes          • Decentralized Mesh Nodes
  • Parametric Insurance Settlement           • Sentinel-1 SAR Automated Ingestion      • On-Chain Smart Contract Relay
```

### 1. "Jev" Standalone Edge Model (System 1 Evolution)
Replace the current Gemini-based JSON triage proxy with **Jev** — a proprietary, sub-10M parameter quantized spiking neural network running locally on CPU in $< 5 \text{ ms}$. This completely isolates the System 1 gate from internet connectivity and external API latency.

### 2. Closed-Loop SCADA & Grid Actuation
Interface AEGIS directly with regional SCADA protocols (IEC 60870-5-104 / DNP3) to autonomously trip circuit breakers and reroute power grids seconds before water reaches transformer bushings, eliminating human operational lag entirely.

### 3. On-Chain Smart Contract Liquidity Relay
Integrate automated EVM / Solana smart contract relays to disburse parametric insurance catastrophe bonds within blocks of physical trigger validation.

---

## Contributors & Acknowledgments

* **Autonomous Disaster Systems Architecture Group**
* Built with pride for high-stakes emergency resilience.
* Hydrodynamic formulations verified against NOAA and IMD storm surge operational criteria.
* Empirical benchmark datasets provided by **Copernicus Emergency Management Service (EMSR357)** and **NOAA IBTrACS**.

---

## License
This project is open-source under the [MIT License](LICENSE). Telemetry datasets and municipal SOP documents conform to National Disaster Management Guidelines.
