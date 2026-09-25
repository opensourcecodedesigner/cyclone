"""
End-to-End Verification of the AEGIS RAG Pipeline
Tests the complete chain:
V-JEPA 2 -> Julia simulation -> Inundation Telemetry -> Insurance Trigger -> LangGraph/LlamaIndex Retrieval -> Gemini Prompt Injection -> Final CAP Dispatch
"""
import os
import sys
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()

# Add project root to sys.path
sys.path.insert(0, os.path.abspath("."))

def test_full_rag_pipeline():
    print("=" * 80)
    print("AEGIS FULL END-TO-END RAG PIPELINE VERIFICATION")
    print("=" * 80)

    # Step 1: V-JEPA Perception telemetry
    from app import get_vjepa2_perception_data, call_julia_physics_engine
    telemetry = get_vjepa2_perception_data()
    phys = telemetry.get("physical_telemetry", {})
    print("\n[STEP 1] V-JEPA 2 Perception Output:")
    print(f"  Land Saturation: {phys.get('land_saturation_index')}")
    print(f"  Manning's n: {phys.get('surface_roughness_manning_n')}")
    print(f"  Friction Multiplier: {phys.get('effective_friction_multiplier')}")

    # Step 2: Julia Simulation Execution
    surge_m = 5.0
    wind_kts = 135.0
    user_iterations = 100
    friction_mult = phys.get("effective_friction_multiplier", 1.30)
    vjepa_iterations = int(round(user_iterations * friction_mult)) # 130 or 156 depending on base
    # If base_iters is scaled from 120 (120*1.30 = 156) or 100 (100*1.30 = 130), let's check
    # In live scenario prompt: "User iterations = 100, V-JEPA-scaled iterations = 156" (from base 120)
    vjepa_iterations = 156
    print(f"\n[STEP 2] Running Julia Hydrodynamic Simulation on port 8080:")
    print(f"  Surge: {surge_m} m | Wind: {wind_kts} kts | Scaled Iterations: {vjepa_iterations}")
    
    vjepa_payload = {
        "land_saturation": phys.get("land_saturation_index", 0.829),
        "surface_roughness_manning_n": phys.get("surface_roughness_manning_n", 0.060),
        "friction_multiplier": friction_mult
    }
    
    success, sim_data, elapsed_ms = call_julia_physics_engine(
        surge_height=surge_m,
        wind_speed_knots=wind_kts,
        iterations=vjepa_iterations,
        vjepa2_perception=vjepa_payload
    )
    assert success, f"Julia call failed: {sim_data}"
    node_results = sim_data["node_results"]
    print(f"  Julia response status: success ({elapsed_ms} ms)")
    print(f"  Effective Diffusion: {sim_data.get('effective_diffusion')}")
    print(f"  Simulated Asset Count: {len(node_results)}")

    # Step 3: Check district_hospital_central
    hosp = next((n for n in node_results if n["id"] == "district_hospital_central"), None)
    assert hosp is not None, "district_hospital_central not found in simulation output"
    hosp_depth = float(hosp["final_water_depth"])
    print(f"\n[STEP 3] Target Asset: district_hospital_central")
    print(f"  Exact Julia Depth: {hosp_depth:.4f} m (expected ~0.4247m)")
    print(f"  Status: {hosp['status']}")
    print(f"  Vulnerability: {hosp['vulnerability_score']}")

    # Step 4: Parametric Insurance Trigger
    from app import evaluate_parametric_insurance_triggers
    triggers = evaluate_parametric_insurance_triggers(node_results)
    hosp_trigger = next((t for t in triggers if t["asset_id"] == "district_hospital_central"), None)
    assert hosp_trigger is not None, "Hospital trigger missing"
    print(f"\n[STEP 4] Parametric Insurance Evaluation:")
    print(f"  Asset: {hosp_trigger['asset_id']}")
    print(f"  Trigger Status: {hosp_trigger['trigger_status']}")
    print(f"  Payout Percentage: {hosp_trigger['payout_percentage']}%")
    print(f"  Policy Condition: {hosp_trigger['policy_condition']}")

    # Step 5: LlamaIndex Semantic Retrieval Test
    print("\n[STEP 5] LlamaIndex Semantic Retrieval (app.py load_local_sop_context):")
    from app import load_local_sop_context, get_sop_vector_index
    index = get_sop_vector_index()
    assert index is not None, "LlamaIndex failed to load vector index"
    
    retriever = index.as_retriever(similarity_top_k=2)
    query_str = "Emergency standard operating procedures for MEDICAL district_hospital_central"
    retrieved_nodes = retriever.retrieve(query_str)
    print(f"  Query: '{query_str}'")
    print(f"  Retrieved Node Count: {len(retrieved_nodes)}")
    for i, node in enumerate(retrieved_nodes):
        file_name = node.node.metadata.get("file_name", "unknown")
        score = getattr(node, "score", None)
        preview = node.node.get_content().strip().replace("\n", " ")[:120]
        print(f"    Rank {i+1} | Source: {file_name} | Score: {score} | Content: {preview}...")

    sop_context = load_local_sop_context("MEDICAL", ["district_hospital_central"])
    print(f"  Retrieved SOP Context Length: {len(sop_context)} characters")

    # Step 6: Full Live Gemini Dispatch Generation with RAG Injection
    print("\n[STEP 6] Executing generate_gemini_dispatch_order with RAG Injection:")
    from app import generate_gemini_dispatch_order
    triage_dec, dispatch_text, insurance_triggers = generate_gemini_dispatch_order(node_results, surge_m, wind_kts)
    
    print("\n" + "=" * 50 + " GENERATED CAP DISPATCH " + "=" * 50)
    print(dispatch_text[:1200] + ("..." if len(dispatch_text) > 1200 else ""))
    print("=" * 124)

    # Step 7: Verify Grounding and Numbers
    print("\n[STEP 7] Grounding & Verification Checks:")
    has_hospital = "district_hospital_central" in dispatch_text or "District Hospital Central" in dispatch_text or "Hospital Central" in dispatch_text
    has_partial = "PARTIAL_PAYOUT_TRIGGER" in dispatch_text or "50%" in dispatch_text
    has_depth = "0.42" in dispatch_text or "0.4247" in dispatch_text
    # SOP mentions: check if SOP specific terms like evacuation, ICU, power isolation, or NDRF appear
    sop_terms = ["vertical", "icu", "evacuat", "de-energiz", "substation", "generator", "oxygen", "ndrf"]
    matched_terms = [t for t in sop_terms if t in dispatch_text.lower()]
    
    print(f"  Hospital asset mentioned: {has_hospital}")
    print(f"  Partial payout / 50% mentioned: {has_partial}")
    print(f"  Water depth (0.42m / 0.4247m) mentioned: {has_depth}")
    print(f"  SOP grounded directives found: {matched_terms}")

    results = {
        "vjepa": telemetry,
        "julia": {
            "hosp_depth": hosp_depth,
            "status": hosp["status"],
            "vulnerability": hosp["vulnerability_score"],
            "effective_diffusion": sim_data.get("effective_diffusion")
        },
        "trigger": hosp_trigger,
        "retrieval": {
            "count": len(retrieved_nodes),
            "sources": [n.node.metadata.get("file_name", "unknown") for n in retrieved_nodes],
            "scores": [getattr(n, "score", None) for n in retrieved_nodes]
        },
        "checks": {
            "has_hospital": has_hospital,
            "has_partial": has_partial,
            "has_depth": has_depth,
            "matched_terms": matched_terms
        },
        "dispatch_snippet": dispatch_text[:600]
    }

    with open("rag_e2e_verification_result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("\nSaved full verification result to rag_e2e_verification_result.json")

if __name__ == "__main__":
    test_full_rag_pipeline()
