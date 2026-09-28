"""
===============================================================================
AEGIS: AI-ASSISTED CYCLONE DISASTER PIPELINE (LANGGRAPH + LLAMA-INDEX + GEMINI)
===============================================================================
State Machine Architecture:
1. Node 1 (System 1 Triage Router): Evaluates hydrodynamic physics telemetry
   and makes rapid binary emergency routing decisions.
2. Conditional Edge (Switch): Emergency -> RAG Retrieval; Safe -> Early Halt.
3. Node 2 (LlamaIndex RAG): Retrieves department-specific Standard Operating
   Procedures (SOPs) from the local Visakhapatnam knowledge base.
4. Node 3 (System 2 Dispatch Synthesizer): Synthesizes physics data and SOPs
   into a structured tactical emergency advisory order for human authorization.
===============================================================================
"""

import os
import sys
import json
import time
from typing_extensions import TypedDict

if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

from dotenv import load_dotenv
load_dotenv()
print("✅ Loaded .env environment variables (load_dotenv)")

from google import genai
from google.genai import types
from langgraph.graph import StateGraph, END
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

# Configure LlamaIndex to use local HuggingFace embeddings (no OpenAI key required)
try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
except ImportError:
    warnings.warn("llama-index-embeddings-huggingface not installed. RAG will use default embeddings.")

# =============================================================================
# STRICT MODE: --strict flag or AEGIS_STRICT=1 makes any fallback a hard error
# =============================================================================
STRICT_MODE = "--strict" in sys.argv or os.getenv("AEGIS_STRICT", "0") == "1"

def _validate_gemini_key():
    """Validate Gemini API key on startup. Fails loudly in strict mode."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        msg = "GEMINI_API_KEY not found in environment. Triage and dispatch will use local fallbacks."
        if STRICT_MODE:
            raise RuntimeError(f"[STRICT MODE] {msg}")
        print(f"\n{'='*75}\n⚠️  WARNING: {msg}\n{'='*75}\n")
        return None
    # Format advisory
    if key.startswith("AIzaSy"):
        print(f"✅ Gemini API key loaded (Google AI Studio format, {len(key)} chars)")
    else:
        print(f"⚠️  Gemini API key loaded but has non-standard format (prefix: {key[:4]}..., {len(key)} chars).")
        print(f"   Standard Google AI Studio keys start with 'AIzaSy'.")
        print(f"   If this is a Vertex AI service-account key, use genai.Client(vertexai=True) instead.")
    return key

_validated_gemini_key = _validate_gemini_key()

# =============================================================================
# 1. PARAMETRIC INSURANCE TRIGGER EVALUATION
# =============================================================================
def evaluate_parametric_insurance_triggers(
    node_results: list,
    full_threshold: float = 1.0,
    partial_threshold: float = 0.3
) -> list:
    """
    Evaluates flood inundation depth against parametric insurance strike thresholds:
    - Depth >= 1.0m: FULL_PAYOUT_TRIGGER (100% instantaneous indemnity liquidity)
    - 0.3m <= Depth < 1.0m: PARTIAL_PAYOUT_TRIGGER (50% emergency operational relief)
    - Depth < 0.3m: NO_TRIGGER (0% payout - within standard retention deductible)
    """
    triggers = []
    for node in node_results:
        depth = float(node.get("final_water_depth", 0.0))
        node_id = str(node.get("id", "unspecified_asset"))
        node_type = str(node.get("type", "infrastructure"))

        if depth >= full_threshold:
            status = "FULL_PAYOUT_TRIGGER"
            payout_pct = 100
        elif depth >= partial_threshold:
            status = "PARTIAL_PAYOUT_TRIGGER"
            payout_pct = 50
        else:
            status = "NO_TRIGGER"
            payout_pct = 0

        triggers.append({
            "asset_id": node_id,
            "asset_type": node_type,
            "flood_depth_m": round(depth, 4),
            "trigger_status": status,
            "payout_percentage": payout_pct,
            "policy_condition": f">= {full_threshold}m (100%) | >= {partial_threshold}m (50%)"
        })
    return triggers

# =============================================================================
# 2. SHARED MEMORY STATE
# =============================================================================
class GraphState(TypedDict, total=False):
    node_results: list
    parametric_triggers: list
    triage_decision: dict
    rag_context: str
    final_dispatch: str

# =============================================================================
# 3. NODE 1: SYSTEM 1 TRIAGE ROUTER (GEMINI FLASH / LANGGRAPH STATE MACHINE)
# =============================================================================
def system1_triage_node(state: GraphState):
    print("\n⚡ [NODE] Running System 1 Triage (LangGraph State Machine)...")
    node_results = state.get("node_results", [])
    api_key = _validated_gemini_key
    
    if not api_key:
        msg = "No Gemini API key — using deterministic Jev Proxy triage instead of Gemini API."
        if STRICT_MODE:
            raise RuntimeError(f"[STRICT MODE] {msg}")
        print(f"\n{'='*75}\n⚠️  FALLBACK: {msg}\n{'='*75}")
        is_emer = any(n.get("status") in ("Critical", "At Risk") for n in node_results)
        dept = "POWER" if any(n.get("type") in ("power_grid", "power") and n.get("status") in ("Critical", "At Risk") for n in node_results) else ("MEDICAL" if is_emer else "NONE")
        decision = {"is_emergency": is_emer, "target_department": dept}
        print(f"   -> Decision (Local Jev Proxy): {decision}")
        return {"triage_decision": decision}

    client = genai.Client(api_key=api_key)
    prompt = """
    Evaluate the JSON payload of flooded infrastructure.
    If ANY node is 'Critical' or 'At Risk', return {"is_emergency": true, "target_department": "APPROPRIATE_DEPT"}.
    Otherwise, return {"is_emergency": false, "target_department": "NONE"}.
    Target department must be one of: "POWER", "MEDICAL", "TRANSPORT", or "NONE".
    """
    response = None
    last_err = None
    candidate_models = ['gemini-3.1-flash-lite', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
    for candidate_model in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=f"{prompt}\n\nDATA:\n{json.dumps(node_results)}",
                    config=types.GenerateContentConfig(response_mime_type="application/json")
                )
                decision = json.loads(response.text)
                if isinstance(decision, list) and len(decision) > 0:
                    decision = decision[0]
                last_err = None
                print(f"   -> System 1 Triage evaluated by {candidate_model}")
                break
            except Exception as err:
                last_err = err
                time.sleep(1.0)
                continue
        if last_err is None:
            break

    if last_err is not None:
        msg = f"Gemini triage API call failed: {last_err}"
        if STRICT_MODE:
            raise RuntimeError(f"[STRICT MODE] {msg}")
        print(f"\n{'='*75}\n⚠️  FALLBACK: {msg}\n   Using deterministic Jev Proxy triage.\n{'='*75}")
        is_emer = any(n.get("status") in ("Critical", "At Risk") for n in node_results)
        decision = {"is_emergency": is_emer, "target_department": "POWER" if is_emer else "NONE"}
    if isinstance(decision, list) and len(decision) > 0:
        decision = decision[0]
    print(f"   -> Decision: {decision}")
    return {"triage_decision": decision}

# =============================================================================
# 4. CONDITIONAL EDGE: THE SWITCH
# =============================================================================
def route_triage(state: GraphState):
    triage = state.get("triage_decision", {})
    if isinstance(triage, list) and len(triage) > 0:
        triage = triage[0]
    if isinstance(triage, dict) and triage.get("is_emergency", False):
        print("   -> 🚨 Emergency Detected! Routing to LlamaIndex RAG...")
        return "retrieve_sop"
    print("   -> ✅ Safe Zone. Halting compute.")
    return END

# =============================================================================
# 5. NODE 2: LLAMAINDEX (THE LIBRARIAN)
# =============================================================================
def retrieve_sop_node(state: GraphState):
    print("📚 [NODE] LlamaIndex retrieving local Visakhapatnam SOPs...")
    dept = state["triage_decision"].get("target_department", "POWER")
    critical_assets = [n.get("id", "") for n in state.get("node_results", []) if n.get("status") in ("Critical", "At Risk")]
    query_str = f"Emergency procedures for {dept} " + " ".join(critical_assets)
    os.makedirs("knowledge_base", exist_ok=True)
    
    rag_text = ""
    try:
        documents = SimpleDirectoryReader("knowledge_base").load_data()
        if not documents:
            raise ValueError("No documents in knowledge_base")
        index = VectorStoreIndex.from_documents(documents)
        retriever = index.as_retriever(similarity_top_k=2)
        
        # Query based on department and specific critical assets
        docs = retriever.retrieve(query_str)
        rag_text = "\n\n".join([d.text for d in docs]) if docs else ""
    except Exception as e:
        msg = f"LlamaIndex vector index failed: {e}"
        if STRICT_MODE:
            raise RuntimeError(f"[STRICT MODE] {msg}")
        print(f"\n{'='*75}\n⚠️  FALLBACK: {msg}\n   Reading SOPs via direct file match instead of vector similarity.\n{'='*75}")
        # Direct retrieval fallback from local SOP documents
        sop_files = [os.path.join("knowledge_base", f) for f in os.listdir("knowledge_base") if f.endswith(('.md', '.txt'))]
        matched_sections = []
        for fpath in sop_files:
            with open(fpath, "r", encoding="utf-8") as f:
                content = f.read()
                if any(asset.lower() in content.lower() for asset in critical_assets if asset) or dept.lower() in content.lower():
                    matched_sections.append(content.strip())
        if matched_sections:
            rag_text = "\n\n".join(matched_sections)
        else:
            rag_text = f"STANDARD OPERATING PROCEDURE FOR {dept}: De-energize high-voltage lines, isolate compromised sectors, and coordinate emergency teams."

    print(f"   -> SOP Context Acquired ({len(rag_text)} characters)")
    return {"rag_context": rag_text}

# =============================================================================
# 6. NODE 3: SYSTEM 2 GEMINI (THE COMMANDER)
# =============================================================================
def system2_dispatch_node(state: GraphState):
    print("🧠 [NODE] Gemini generating final tactical dispatch...")
    node_results = state.get("node_results", [])
    triggers = state.get("parametric_triggers")
    if not triggers and node_results:
        triggers = evaluate_parametric_insurance_triggers(node_results)

    # Strictly build exact, unparaphrased telemetry and parametric trigger blocks
    parametric_block = "[PARAMETRIC INSURANCE TRIGGER STATUS]\n" + "\n".join([
        f"• {t['asset_id']:<35} : {t['trigger_status']} ({t['payout_percentage']}% Payout @ {t['flood_depth_m']}m)"
        for t in triggers
    ])

    threat_block = "[CRITICAL THREAT EVALUATION]\n" + "\n".join([
        f"- {n.get('id')}: {float(n.get('final_water_depth', 0.0)):.4f}m inundation. Vulnerability: {n.get('vulnerability_score')} ({n.get('status')})."
        for n in node_results
    ])

    api_key = _validated_gemini_key
    if not api_key:
        msg = "No Gemini API key — using deterministic CAP template instead of Gemini synthesis."
        if STRICT_MODE:
            raise RuntimeError(f"[STRICT MODE] {msg}")
        print(f"\n{'='*75}\n⚠️  FALLBACK: {msg}\n{'='*75}")
        final_dispatch = (
            "=== AEGIS COMMON ALERTING PROTOCOL (CAP) TACTICAL DISPATCH ADVISORY ===\n"
            "INCIDENT: CYCLONIC SURGE INUNDATION & INFRASTRUCTURE FAILURE\n"
            "SEVERITY: EXTREME | URGENCY: IMMEDIATE | CERTAINTY: OBSERVED\n\n"
            f"{threat_block}\n\n"
            f"{parametric_block}\n\n"
            "[MANDATORY ACTION DIRECTIVES]\n"
            f"{state.get('rag_context', 'Isolate compromised power assets, vertically evacuate healthcare facilities, and redirect traffic.')}"
        )
        return {"final_dispatch": final_dispatch, "parametric_triggers": triggers}

    client = genai.Client(api_key=api_key)
    prompt = f"""
    Write an authoritative emergency dispatch order conforming to Common Alerting Protocol (CAP) standards.
    Use the Physics Telemetry: {json.dumps(node_results)}
    Strictly follow this Local SOP: {state['rag_context']}

    CRITICAL INSTRUCTION: Include the EXACT verbatim [PARAMETRIC INSURANCE TRIGGER STATUS] block below without modifying or rounding any depth numbers:
    {parametric_block}
    """
    response = None
    last_err = None
    candidate_models = ['gemini-3.1-flash-lite', 'gemini-3.5-flash-lite', 'gemini-3.5-flash']
    for candidate_model in candidate_models:
        for attempt in range(2):
            try:
                response = client.models.generate_content(
                    model=candidate_model,
                    contents=prompt
                )
                final_dispatch = response.text
                # Enforce exact block if altered by LLM
                if "[PARAMETRIC INSURANCE TRIGGER STATUS]" not in final_dispatch:
                    final_dispatch += f"\n\n{parametric_block}"
                last_err = None
                print(f"   -> System 2 Dispatch generated by {candidate_model}")
                break
            except Exception as err:
                last_err = err
                time.sleep(1.0)
                continue
        if last_err is None:
            break

    if last_err is not None:
        msg = f"Gemini dispatch API call failed: {last_err}"
        if STRICT_MODE:
            raise RuntimeError(f"[STRICT MODE] {msg}")
        print(f"\n{'='*75}\n⚠️  FALLBACK: {msg}\n   Generating deterministic CAP order.\n{'='*75}")
        final_dispatch = (
            "=== AEGIS COMMON ALERTING PROTOCOL (CAP) TACTICAL DISPATCH ADVISORY ===\n"
            "INCIDENT: CYCLONIC SURGE INUNDATION & INFRASTRUCTURE FAILURE\n"
            "SEVERITY: EXTREME | URGENCY: IMMEDIATE | CERTAINTY: OBSERVED\n\n"
            f"{threat_block}\n\n"
            f"{parametric_block}\n\n"
            "[MANDATORY ACTION DIRECTIVES]\n"
            f"{state.get('rag_context', 'Isolate compromised power assets, vertically evacuate healthcare facilities, and redirect traffic.')}"
        )

    return {"final_dispatch": final_dispatch, "parametric_triggers": triggers}

# =============================================================================
# 6. BUILD THE LANGGRAPH STATE MACHINE
# =============================================================================
workflow = StateGraph(GraphState)

# Add Nodes
workflow.add_node("triage", system1_triage_node)
workflow.add_node("retrieve_sop", retrieve_sop_node)
workflow.add_node("dispatch", system2_dispatch_node)

# Connect Edges
workflow.set_entry_point("triage")
workflow.add_conditional_edges("triage", route_triage)
workflow.add_edge("retrieve_sop", "dispatch")
workflow.add_edge("dispatch", END)

# Compile Pipeline
aegis_pipeline = workflow.compile()

# =============================================================================
# 7. EXECUTION / TEST ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    # Test with the exact output your Julia engine just calculated
    test_data = [
        {"id": "power_substation_alpha", "final_water_depth": 2.5391, "status": "Critical"},
        {"id": "district_hospital_central", "final_water_depth": 1.9391, "status": "Critical"}
    ]
    
    # Query active Julia microservice if '--live' is specified
    if "--live" in sys.argv:
        import requests
        try:
            print("Querying active Julia simulation microservice (http://localhost:8080/simulate_surge)...")
            res = requests.post("http://localhost:8080/simulate_surge", json={"surge_height": 8.5, "wind_speed": 150}, timeout=10)
            if res.status_code == 200:
                live_nodes = res.json().get("node_results", [])
                if live_nodes:
                    test_data = live_nodes
                    print(f"Loaded {len(test_data)} live nodes from Julia physics engine.")
        except Exception as err:
            print(f"Note: Could not reach Julia server ({err}), using simulation output benchmark.")

    print("Starting AEGIS Decision Pipeline (Human Authorization Required)...")
    final_state = aegis_pipeline.invoke({"node_results": test_data})
    
    if "final_dispatch" in final_state:
        print("\n=== FINAL DISPATCH ORDER ===")
        print(final_state["final_dispatch"])

    if "parametric_triggers" in final_state:
        print("\n=== PARAMETRIC INSURANCE TRIGGER STATUS ===")
        for trig in final_state["parametric_triggers"]:
            print(f"• {trig['asset_id']:<35} : {trig['trigger_status']} ({trig['payout_percentage']}% Payout @ {trig['flood_depth_m']}m)")
