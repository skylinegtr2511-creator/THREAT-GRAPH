from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Define the Directed Acyclic Graph (DAG) structure
model = DiscreteBayesianNetwork([
    ('Reconnaissance', 'Exploitation'),
    ('Exploitation', 'Backdoor'),
    ('Backdoor', 'Impact')
])

# 2. Define Conditional Probability Tables (CPTs)
cpd_recon = TabularCPD(variable='Reconnaissance', variable_card=2, 
                       values=[[0.80],   
                               [0.20]])  

cpd_exploit = TabularCPD(variable='Exploitation', variable_card=2, 
                         values=[[0.99, 0.40],  
                                 [0.01, 0.60]], 
                         evidence=['Reconnaissance'], evidence_card=[2])

cpd_backdoor = TabularCPD(variable='Backdoor', variable_card=2, 
                          values=[[0.95, 0.30],  
                                  [0.05, 0.70]], 
                          evidence=['Exploitation'], evidence_card=[2])

cpd_impact = TabularCPD(variable='Impact', variable_card=2, 
                        values=[[0.99, 0.10],  
                                [0.01, 0.90]], 
                        evidence=['Backdoor'], evidence_card=[2])

# 3. Add CPTs to the model and validate
model.add_cpds(cpd_recon, cpd_exploit, cpd_backdoor, cpd_impact)

if model.check_model():
    print("DAG and CPTs are mathematically valid.\n")

# 4. Exact Inference: Calculate Posterior Probabilities
inference = VariableElimination(model)

print("--- Baseline Risk (No active alerts) ---")
baseline_risk = inference.query(variables=['Impact'])
print(baseline_risk)

print("\n--- Posterior Risk (IDS detects Exploitation) ---")
posterior_risk = inference.query(variables=['Impact'], evidence={'Exploitation': 1})
print(posterior_risk)