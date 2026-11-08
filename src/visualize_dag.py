import networkx as nx
import matplotlib.pyplot as plt
from dynamic_model import model 
import os

os.makedirs("docs", exist_ok=True)

# Extract edges from your pgmpy model
G = nx.DiGraph(model.edges())

plt.figure(figsize=(12, 8))
# Use a bipartite or multipartite layout for time slices if desired, 
# but spring_layout works well for general DAGs
pos = nx.spring_layout(G, seed=42) 

nx.draw(G, pos, with_labels=True, node_size=3500, node_color="#4da6ff", 
        font_size=10, font_weight="bold", arrows=True, arrowsize=20)

plt.title("2-Time-Slice Dynamic Bayesian Network (2TBN)", fontsize=16)
plt.savefig(os.path.join("docs", "DAG_visualization.png"), format="PNG")
print("DAG visualization saved to docs/DAG_visualization.png")