# THREAT-GRAPH: Multi-Stage Cyber-Attack Path Prediction and Risk Inference

**Authors:** Karthikeya Kosuri, U Navaneeth and G Surya Tejesh  
**Institution:** Amrita Vishwa Vidyapeetham  
**Course:** 22AIE301 (Probabilistic Reasoning)  

## Overview
THREAT-GRAPH is a probabilistic cybersecurity inference framework designed to predict multi-stage cyber-attack paths and evaluate system risk. By converting static cybersecurity frameworks (such as MITRE ATT&CK) into dynamic mathematical models, the system computes the real-time posterior probabilities of a network breach. It tracks how threats propagate through a system over time, allowing Security Operations Center (SOC) analysts to forecast future states based on current intrusion detection evidence.

## Mathematical Foundation
The project relies on rigorous probabilistic and graphical modeling techniques:
*   **Directed Acyclic Graphs (DAGs):** Used to map the sequential steps an attacker must take to compromise a system (e.g., Reconnaissance -> Exploitation -> Backdoor).
*   **Conditional Probability Tables (CPTs):** Matrices that quantify the conditional dependencies between attack stages.
*   **2-Time-Slice Dynamic Bayesian Networks (2TBN):** Unrolled Markovian networks that model state changes across discrete time steps (T0 to T1), enabling future state prediction.
*   **Exact Inference:** Utilizes Variable Elimination (Belief Propagation) to continuously calculate the posterior probability of a successful exploit as new evidence is introduced.

## Data Sources
To avoid arbitrary probability assignments, the network's transition matrices are grounded in empirical data:
1.  **CISA Known Exploited Vulnerabilities (KEV):** Used to map real-world exploit availability and patch statuses.
2.  **UNSW-NB15 Dataset:** A comprehensive network intrusion dataset used to extract the raw statistical frequencies of multi-stage attack behaviors (Reconnaissance, Exploits, Backdoors, and DoS) to populate the baseline CPTs.

## Features
*   **Static Threat Modeling:** Baseline Bayesian network inference using `pgmpy`.
*   **Dynamic Predictive Inference:** Cross-time-slice forecasting using a 2TBN.
*   **SOC Dashboard:** An interactive web application built with Streamlit and Plotly to visualize probability distributions dynamically.
*   **Custom Network Builder:** A UI tool that allows users to construct custom DAG topologies, validates them for acyclic properties using `networkx`, and auto-generates mathematically sound CPT matrices for arbitrary graph structures.

## Project Structure
```text
THREAT-GRAPH/
├── data/                           
│   ├── cisa_kev.json               # CISA vulnerabilities dataset
│   └── unsw_train.csv              # UNSW-NB15 training data 
├── src/                            
│   ├── fetch_dataset.py            # Script to ingest CISA KEV data
│   ├── fetch_unsw.py               # Script to ingest UNSW-NB15 data
│   ├── calculate_probabilities.py  # Pandas script to extract CPT stats from UNSW data
│   ├── threat_model.py             # Static Bayesian Network DAG and exact inference
│   ├── dynamic_model.py            # The 2-Time-Slice Dynamic Bayesian Network (2TBN)
│   └── visualize_dag.py            # Generates DAG topological mapping
├── docs/                           
│   └── DAG_visualization.png       # Visual output of the Bayesian Network nodes
├── app.py                          # Streamlit SOC Dashboard and Custom Network Builder
├── requirements.txt                # Python dependencies
└── README.md                       # Project documentation