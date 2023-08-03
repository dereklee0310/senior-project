import cirq
import numpy as np

qubits = cirq.GridQubit.square(2)
print(qubits)
circuit = cirq.Circuit()
circuit.append([cirq.H(qubit) for qubit in qubits])
circuit.append(cirq.CCNOT(qubits[1], qubits[2], qubits[3]))
circuit.append(cirq.CCNOT(qubits[0], qubits[1], qubits[2]))
circuit.append(cirq.I(qubits[3])) # formatting

circuit.append([cirq.rx(np.pi)(qubit) for qubit in qubits])
circuit.append(cirq.CCNOT(qubits[1], qubits[0], qubits[3]))
circuit.append(cirq.I(qubits[2])) # formatting

circuit.append([cirq.ry(np.pi)(qubit) for qubit in qubits])
circuit.append(cirq.CCNOT(qubits[2], qubits[3], qubits[0]))
circuit.append(cirq.I(qubits[1])) # formatting
circuit.append([cirq.rz(np.pi)(qubit) for qubit in qubits])

print(circuit)
