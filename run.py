import json
import numpy as np
import person1_qubo as p1
import person2_qaoa as p2
from qiskit.qasm3 import dumps

def main():
    # Scenario R parameters
    sc_R = {
        'cap': 5, 'S': 7, 'wo': 4, 'ws': 0.5, 'link': 'EW', 'fr': 0.6,
        'A_NS': [3, 6, 6, 3], 'A_EW': [5, 1, 1, 2],
        'B_NS': [3, 5, 4, 3], 'B_EW': [2, 6, 4, 6]
    }
    
    print("--- 1. Building QUBO (Person 1) ---")
    Q, offset = p1.build_qubo(sc_R)
    p1.export_outputs(sc_R)
    
    print("--- 2. Running QAOA Optimization & Sampling 4096 Shots (Person 2) ---")
    p = 2  # Circuit depth
    counts = p2.run_qaoa(Q, offset, p=p, num_restarts=5)
    
    # Export counts.json
    with open("counts.json", "w") as f:
        json.dump(counts, f, indent=4)
    print("Exported counts.json successfully!")
    
    # Export circuit.qasm (OpenQASM 3 format)
    qc_unmeasured = p2.build_qaoa_circuit(Q, p, np.zeros(2 * p))
    qasm_text = dumps(qc_unmeasured)
    with open("circuit.qasm", "w") as f:
        f.write(qasm_text)
    print("Exported circuit.qasm successfully!")
    
    # Find best sampled schedule from counts
    best_bitstring = max(counts, key=counts.get)
    best_bits = [int(c) for c in best_bitstring]
    best_cost = p1.official_cost(best_bits, sc_R)
    
    print(f"\n--- Pipeline Complete ---")
    print(f"Top Sampled Schedule: {best_bitstring} with Official Cost: {best_cost}")
    print(f"Total Shots Accounted For: {sum(counts.values())} (Target: 4096)")

if __name__ == "__main__":
    main()