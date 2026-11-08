import streamlit as st
import plotly.express as px
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
import warnings
import numpy as np

warnings.filterwarnings("ignore", category=FutureWarning)

st.set_page_config(page_title="THREAT-GRAPH | AI Cyber Risk", page_icon="🛡️", layout="wide")

# --- Initialize Session State for Custom Network Builder ---
if 'custom_nodes' not in st.session_state:
    st.session_state.custom_nodes = []
if 'custom_edges' not in st.session_state:
    st.session_state.custom_edges = []

@st.cache_resource
def initialize_inference_engine():
    model = DiscreteBayesianNetwork([
        ('Recon_T0', 'Exploit_T0'), ('Exploit_T0', 'Backdoor_T0'),
        ('Recon_T1', 'Exploit_T1'), ('Exploit_T1', 'Backdoor_T1'),
        ('Recon_T0', 'Recon_T1'), ('Exploit_T0', 'Exploit_T1'), ('Backdoor_T0', 'Backdoor_T1')
    ])
    cpd_recon_t0 = TabularCPD('Recon_T0', 2, [[0.9121], [0.0879]])
    cpd_exploit_t0 = TabularCPD('Exploit_T0', 2, [[0.90, 0.40], [0.10, 0.60]], evidence=['Recon_T0'], evidence_card=[2])
    cpd_backdoor_t0 = TabularCPD('Backdoor_T0', 2, [[0.99, 0.85], [0.01, 0.15]], evidence=['Exploit_T0'], evidence_card=[2])
    cpd_recon_t1 = TabularCPD('Recon_T1', 2, [[0.95, 0.20], [0.05, 0.80]], evidence=['Recon_T0'], evidence_card=[2])
    cpd_exploit_t1 = TabularCPD('Exploit_T1', 2, [[0.99, 0.60, 0.70, 0.10], [0.01, 0.40, 0.30, 0.90]], evidence=['Exploit_T0', 'Recon_T1'], evidence_card=[2, 2])
    cpd_backdoor_t1 = TabularCPD('Backdoor_T1', 2, [[0.99, 0.70, 0.80, 0.10], [0.01, 0.30, 0.20, 0.90]], evidence=['Backdoor_T0', 'Exploit_T1'], evidence_card=[2, 2])
    model.add_cpds(cpd_recon_t0, cpd_exploit_t0, cpd_backdoor_t0, cpd_recon_t1, cpd_exploit_t1, cpd_backdoor_t1)
    return VariableElimination(model)

inference = initialize_inference_engine()

# --- Top Navigation ---
st.title("🛡️ THREAT-GRAPH: Probabilistic Cybersecurity Framework")
tab1, tab2 = st.tabs(["📊 SOC Dashboard (2TBN Inference)", "🏗️ Custom Network Builder"])

# ==========================================
# TAB 1: SOC DASHBOARD (Original App)
# ==========================================
with tab1:
    st.markdown("Real-time posterior breach probability calculator using the predefined 2-Time-Slice DAG.")
    
    col_input, col_metrics = st.columns([1, 3])
    
    with col_input:
        st.subheader("SOC Alerts (T0)")
        recon_alert = st.toggle("🚨 Recon Detected", value=False)
        exploit_alert = st.toggle("🚨 Exploit Detected", value=False)
        backdoor_alert = st.toggle("🚨 Backdoor Detected", value=False)
        
        evidence = {}
        if recon_alert: evidence['Recon_T0'] = 1
        if exploit_alert: evidence['Exploit_T0'] = 1
        if backdoor_alert: evidence['Backdoor_T0'] = 1

    with col_metrics:
        try:
            if evidence:
                res_recon = inference.query(variables=['Recon_T1'], evidence=evidence).values[1]
                res_exploit = inference.query(variables=['Exploit_T1'], evidence=evidence).values[1]
                res_backdoor = inference.query(variables=['Backdoor_T1'], evidence=evidence).values[1]
            else:
                res_recon = inference.query(variables=['Recon_T1']).values[1]
                res_exploit = inference.query(variables=['Exploit_T1']).values[1]
                res_backdoor = inference.query(variables=['Backdoor_T1']).values[1]
                
            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Recon (T1)", f"{res_recon * 100:.1f}%")
            m2.metric("Predicted Exploit (T1)", f"{res_exploit * 100:.1f}%")
            m3.metric("Predicted Backdoor (T1)", f"{res_backdoor * 100:.1f}%")
            
            df_probs = pd.DataFrame({
                'Stage': ['Reconnaissance', 'Exploitation', 'Backdoor'],
                'Probability': [res_recon, res_exploit, res_backdoor]
            })
            fig = px.bar(df_probs, x='Probability', y='Stage', orientation='h', range_x=[0, 1], text_auto='.1%', color='Probability', color_continuous_scale=['#00CC96', '#FFA15A', '#EF553B'])
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        except Exception as e:
            st.error("Error computing probabilities.")

# ==========================================
# TAB 2: CUSTOM NETWORK BUILDER
# ==========================================
with tab2:
    st.markdown("Dynamically define Directed Acyclic Graph (DAG) structures. The system auto-generates mathematically valid CPTs for custom inference.")
    
    build_col1, build_col2 = st.columns([1, 2])
    
    with build_col1:
        st.subheader("1. Define States (Nodes)")
        new_node = st.text_input("Node Name (e.g., Phishing)")
        if st.button("Add Node", use_container_width=True):
            if new_node and new_node not in st.session_state.custom_nodes:
                st.session_state.custom_nodes.append(new_node)
                # Clear any compiled model when topology changes
                st.session_state.pop('custom_inference', None) 
                st.rerun()
                
        st.subheader("2. Define Dependencies (Edges)")
        if len(st.session_state.custom_nodes) >= 2:
            src = st.selectbox("Parent Node", st.session_state.custom_nodes, key="src")
            dst = st.selectbox("Child Node", st.session_state.custom_nodes, key="dst")
            
            if st.button("Add Directed Edge", use_container_width=True):
                if src != dst and (src, dst) not in st.session_state.custom_edges:
                    st.session_state.custom_edges.append((src, dst))
                    st.session_state.pop('custom_inference', None)
                    st.rerun()
        else:
            st.info("Add at least 2 nodes to create edges.")
            
        if st.button("🗑️ Clear Network", type="primary", use_container_width=True):
            st.session_state.custom_nodes = []
            st.session_state.custom_edges = []
            st.session_state.pop('custom_inference', None)
            st.rerun()

    with build_col2:
        st.subheader("Interactive Graph Topology")
        
        G = nx.DiGraph()
        if st.session_state.custom_nodes:
            G.add_nodes_from(st.session_state.custom_nodes)
            G.add_edges_from(st.session_state.custom_edges)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            pos = nx.spring_layout(G, seed=42)
            is_dag = nx.is_directed_acyclic_graph(G)
            node_color = "#4da6ff" if is_dag else "#ff4d4d"
            
            nx.draw(G, pos, with_labels=True, node_size=3000, node_color=node_color, 
                    font_size=10, font_weight="bold", arrows=True, arrowsize=20, ax=ax)
            st.pyplot(fig)
            
            if is_dag:
                st.success("✅ Valid Directed Acyclic Graph (DAG)")
                
                # --- AUTO-GENERATE CPTs & COMPILE MODEL ---
                if st.button("⚙️ Compile Custom Network & Generate CPTs", use_container_width=True):
                    try:
                        custom_model = DiscreteBayesianNetwork(st.session_state.custom_edges)
                        # Add nodes that might not have edges yet
                        custom_model.add_nodes_from(st.session_state.custom_nodes)
                        
                        for node in custom_model.nodes():
                            parents = list(custom_model.predecessors(node))
                            num_parents = len(parents)
                            
                            if num_parents == 0:
                                # Root node: Uniform 50/50 baseline probability
                                cpd = TabularCPD(variable=node, variable_card=2, values=[[0.5], [0.5]])
                            else:
                                # Child node: Generate valid random distributions for 2^N combinations
                                cols = 2 ** num_parents
                                state_0 = np.random.uniform(0.1, 0.9, size=(1, cols)) # Prevent absolute 0s or 1s
                                state_1 = 1.0 - state_0
                                values = np.vstack((state_0, state_1))
                                cpd = TabularCPD(variable=node, variable_card=2, 
                                                 values=values, 
                                                 evidence=parents, evidence_card=[2]*num_parents)
                            custom_model.add_cpds(cpd)
                            
                        custom_model.check_model()
                        st.session_state.custom_inference = VariableElimination(custom_model)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Failed to compile model: {str(e)}")
            else:
                st.error("❌ Invalid Topology: Cycle detected. Bayesian Networks must be acyclic.")

    # --- DYNAMIC INFERENCE DASHBOARD ---
    if 'custom_inference' in st.session_state:
        st.divider()
        st.subheader("🎯 Custom Network Inference Dashboard")
        
        inf_col1, inf_col2 = st.columns([1, 2])
        
        with inf_col1:
            st.markdown("**Inject Evidence (Observed States)**")
            custom_evidence = {}
            for node in st.session_state.custom_nodes:
                # Dynamically create toggles for whatever nodes the user built
                if st.toggle(f"🚨 {node} Detected", key=f"tog_{node}"):
                    custom_evidence[node] = 1
                    
        with inf_col2:
            st.markdown("**Query Target Probability**")
            # You can only query nodes that aren't currently being set as evidence
            available_targets = [n for n in st.session_state.custom_nodes if n not in custom_evidence]
            
            if available_targets:
                target_node = st.selectbox("Select node to predict:", available_targets)
                
                try:
                    if custom_evidence:
                        result = st.session_state.custom_inference.query(variables=[target_node], evidence=custom_evidence)
                    else:
                        result = st.session_state.custom_inference.query(variables=[target_node])
                        
                    prob_true = result.values[1]
                    
                    st.metric(label=f"Posterior Probability of {target_node}=True", 
                              value=f"{prob_true * 100:.1f}%",
                              delta="Elevated" if prob_true > 0.5 else "Nominal", 
                              delta_color="inverse")
                except Exception as e:
                    st.warning("Query failed. Ensure the graph is fully connected.")
            else:
                st.info("All nodes are set as evidence. Nothing left to predict.")