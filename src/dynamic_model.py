from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define the 2-Time-Slice Unrolled DAG Structure
# T0 represents the current time slice; T1 represents the future time slice.
model = DiscreteBayesianNetwork([
    # Time Slice 0 (Initial Attack State)
    ('Recon_T0', 'Exploit_T0'),
    ('Exploit_T0', 'Backdoor_T0'),
    
    # Time Slice 1 (Future Attack State)
    ('Recon_T1', 'Exploit_T1'),
    ('Exploit_T1', 'Backdoor_T1'),
    
    # Inter-Slice Temporal Edges (Markov State Transitions: T0 -> T1)
    ('Recon_T0', 'Recon_T1'),
    ('Exploit_T0', 'Exploit_T1'),
    ('Backdoor_T0', 'Backdoor_T1')
])

# 2. Define Initial State CPTs for T0 (Anchored by our extracted UNSW-NB15 data)
# P(Recon) = 8.79% baseline
cpd_recon_t0 = TabularCPD('Recon_T0', 2, [[0.9121], [0.0879]])

# Exploitation given Reconnaissance at T0
cpd_exploit_t0 = TabularCPD('Exploit_T0', 2, 
                            [[0.90, 0.40], 
                             [0.10, 0.60]], 
                            evidence=['Recon_T0'], evidence_card=[2])
                            
# Backdoor given Exploitation at T0
cpd_backdoor_t0 = TabularCPD('Backdoor_T0', 2, 
                             [[0.99, 0.85], 
                              [0.01, 0.15]], 
                             evidence=['Exploit_T0'], evidence_card=[2])

# 3. Define Temporal Transition CPTs for T1
# State persistence: How likely is Recon to continue in T1 if active in T0?
cpd_recon_t1 = TabularCPD('Recon_T1', 2, 
                          [[0.95, 0.20], 
                           [0.05, 0.80]], 
                          evidence=['Recon_T0'], evidence_card=[2])
                          
# Exploitation at T1 depends on BOTH Exploitation at T0 and Recon at T1
cpd_exploit_t1 = TabularCPD('Exploit_T1', 2, 
                            [[0.99, 0.60, 0.70, 0.10], 
                             [0.01, 0.40, 0.30, 0.90]], 
                            evidence=['Exploit_T0', 'Recon_T1'], evidence_card=[2, 2])
                            
# Backdoor at T1 depends on BOTH Backdoor at T0 and Exploitation at T1
cpd_backdoor_t1 = TabularCPD('Backdoor_T1', 2, 
                             [[0.99, 0.70, 0.80, 0.10], 
                              [0.01, 0.30, 0.20, 0.90]], 
                             evidence=['Backdoor_T0', 'Exploit_T1'], evidence_card=[2, 2])

# 4. Initialize and Validate the Multi-Stage Network
model.add_cpds(cpd_recon_t0, cpd_exploit_t0, cpd_backdoor_t0, 
               cpd_recon_t1, cpd_exploit_t1, cpd_backdoor_t1)

if model.check_model():
    print("2TBN Structure and Temporal Transition Matrices Validated.\n")

# 5. Exact Inference: Cross-Time-Slice Prediction
inference = VariableElimination(model)

print("--- Predictive Cyber Inference: Forecasting T1 based on T0 ---")
print("Scenario: An Intrusion Detection System flags Reconnaissance at Time 0.")
print("Querying the posterior probability of a Backdoor breach at Time 1...\n")

posterior = inference.query(variables=['Backdoor_T1'], evidence={'Recon_T0': 1})
print(posterior)