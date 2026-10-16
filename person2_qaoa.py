import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.primitives import StatevectorSampler
from scipy.optimize import minimize

def build_qaoa_circuit(Q, p, params):
    """
    Builds the 8-qubit QAOA circuit for a given QUBO matrix Q and depth p.
    """
    num_qubits = Q.shape[0]
    qc = QuantumCircuit(num_qubits)
    
    gammas = params[:p]
    betas = params[p:]
    
    # Initial state: Equal superposition on all 8 qubits
    qc.h(range(num_qubits))
    
    for layer in range(p):
        gamma = gammas[layer]
        beta = betas[layer]
        
        # 1. Cost Layer
        for i in range(num_qubits):
            if Q[i, i] != 0:
                qc.rz(2.0 * gamma * Q[i, i], i)
                
            for j in range(i + 1, num_qubits):
                if Q[i, j] != 0:
                    qc.rzz(2.0 * gamma * Q[i, j], i, j)
                    
        # 2. Mixer Layer
        for i in range(num_qubits):
            qc.rx(2.0 * beta, i)
            
    return qc

def compute_expected_cost(params, Q, offset, p):
    """
    Evaluates the expected cost of the QAOA circuit given current parameters.
    """
    qc = build_qaoa_circuit(Q, p, params)
    sv = Statevector.from_instruction(qc)
    probs = sv.probabilities_dict()
    
    total_cost = 0.0
    for bitstring, prob in probs.items():
        b_str = bitstring[::-1]
        x = np.array([int(char) for char in b_str])
        cost = float(x @ Q @ x + offset)
        total_cost += cost * prob
        
    return total_cost

def run_qaoa(Q, offset, p=3, num_restarts=5):
    """
    Multi-start QAOA training using COBYLA across multiple random initializations 
    to avoid local minima, followed by 4096-shot sampling.
    """
    best_params = None
    best_expected_cost = float('inf')
    
    print(f"Running multi-start QAOA optimization ({num_restarts} restarts, depth p={p})...")
    
    # Multi-start loop to find the best parameter set
    for seed in range(num_restarts):
        np.random.seed(seed * 10)
        initial_params = np.random.uniform(0, np.pi, 2 * p)
        
        result = minimize(
            compute_expected_cost,
            initial_params,
            args=(Q, offset, p),
            method='COBYLA',
            options={'maxiter': 400, 'disp': False}
        )
        
        if result.fun < best_expected_cost:
            best_expected_cost = result.fun
            best_params = result.x
            
    print(f"Optimization complete. Best expected cost found: {best_expected_cost:.2f}")
    
    # Build final optimized circuit using the best parameters found
    qc_final = build_qaoa_circuit(Q, p, best_params)
    qc_final.measure_all()
    
    # Sample 4096 shots
    sampler = StatevectorSampler()
    job = sampler.run([(qc_final,)], shots=4096)
    result_pub = job.result()[0]
    
    counts_raw = result_pub.data.meas.get_counts()
    counts = {str(k)[::-1]: int(v) for k, v in counts_raw.items()}
    return counts