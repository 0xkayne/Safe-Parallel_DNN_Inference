import os
import networkx as nx
from loader import ModelLoader

DATASET_DIR = "datasets_260120"

def diagnose():
    csv_files = [f for f in os.listdir(DATASET_DIR) if f.endswith('.csv')]
    for csv_file in sorted(csv_files):
        path = os.path.join(DATASET_DIR, csv_file)
        print(f"Checking {csv_file}...")
        try:
            G, _ = ModelLoader.load_model_from_csv(path)
            is_dag = nx.is_directed_acyclic_graph(G)
            print(f"  - Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
            print(f"  - Is DAG: {is_dag}")
            if not is_dag:
                cycles = list(nx.simple_cycles(G))
                print(f"  - ❌ CYCLE DETECTED! First cycle: {cycles[0][:5]}...")
        except Exception as e:
            print(f"  - ❌ Error loading: {str(e)}")

if __name__ == "__main__":
    diagnose()
