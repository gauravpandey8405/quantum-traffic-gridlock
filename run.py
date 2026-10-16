import argparse
import json
import os
import numpy as np
import person1_qubo as p1
import person2_qaoa as p2
from qiskit.qasm3 import dumps

def main():
    parser = argparse.ArgumentParser(description="Quantum Traffic Gridlock QAOA Pipeline")
    parser.add_argument("--scenario", type=str, required=True, help="Path to scenario JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--out", type=str, default="./", help="Output directory")
    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.out, exist_ok=True)
    
    # Load scenario JSON
    with open(args.scenario, "r") as f:
        sc = json.load(f)
        
    print(f"--- Loaded Scenario from {args.scenario} ---")
    
    print("--- 1. Building QUBO (Person 1) ---")
    Q, offset = p1.build_qubo(sc)
    
    # Export qubo.json and schedule.json to specified output directory
    qubo_path = os.path.join(args.out, "qubo.json")
    schedule_path = os.path.join(args.out, "schedule.json")
    
    qubo_data = {"Q": Q.tolist(), "offset": offset}
    with open(qubo_path, "w") as f:
        json.dump(qubo_data, f, indent=4)
        
    costs = p1.official_costs(sc)
    elite = p1.elite_set(costs, 13)
    schedule_data = {
        "best_index": int(elite[0]),
        "best_cost": float(costs[elite[0]]),
        "random_average": float(costs.mean()),
        "elite_indices": elite
    }
    with open(schedule_path, "w") as f:
        json.dump(schedule_data, f, indent=4)
    print(f"Exported {qubo_path} and {schedule_path} successfully!")
    
    print(f"--- 2. Running QAOA Optimization & Sampling (Seed: {args.seed}) ---")
    p = 2
    counts = p2.run_qaoa(Q, offset, p=p, num_restarts=5)
    
    counts_path = os.path.join(args.out, "counts.json")
    with open(counts_path, "w") as f:
        json.dump(counts, f, indent=4)
    print(f"Exported {counts_path} successfully!")
    
    qasm_path = os.path.join(args.out, "circuit.qasm")
    qc_unmeasured = p2.build_qaoa_circuit(Q, p, np.zeros(2 * p))
    with open(qasm_path, "w") as f:
        f.write(dumps(qc_unmeasured))
    print(f"Exported {qasm_path} successfully!")
    
    best_bitstring = max(counts, key=counts.get)
    best_bits = [int(c) for c in best_bitstring]
    best_cost = p1.official_cost(best_bits, sc)
    
    print(f"\n--- Pipeline Complete ---")
    print(f"Top Sampled Schedule: {best_bitstring} with Official Cost: {best_cost}")

if __name__ == "__main__":
    main()