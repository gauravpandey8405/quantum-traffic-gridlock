from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import qiskit.qasm3

# 1. Build a 2-qubit Bell circuit
qc = QuantumCircuit(2)
qc.h(0)        # Put qubit 0 into 50/50 superposition
qc.cx(0, 1)    # Entangle qubit 0 and qubit 1

# 2. Compute exact Statevector probabilities
sv = Statevector.from_instruction(qc)
print("Bell State Probabilities:", sv.probabilities_dict())

# 3. Export to OpenQASM 3 text
print("\n--- OpenQASM 3 Output ---")
print(qiskit.qasm3.dumps(qc))
