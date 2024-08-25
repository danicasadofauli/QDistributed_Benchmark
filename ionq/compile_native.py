import qiskit.circuit.random
from .ionq_gates import MSGate, GPIGate, GPI2Gate
import numpy as np
from qiskit import QuantumCircuit, transpile


def compile_to_ionq_native_gates(qc, check=False):
    qc_new = transpile(qc, basis_gates=["id", "rx", "ry", "rz", "cx"])
    qc_new = cnot_to_msgate(qc_new, check=check)
    qc_new = rx_ry_to_rz(qc_new, check=check)
    qc_new = rx_ry_trivial_phases_to_gpigates(qc_new, check=check)
    qc_new = consume_rz_gates(qc_new, check=check)
    return qc_new


def check_qc_compilation(qc1, qc2):
    qc1_unitary = qiskit.quantum_info.Operator(qc1).data
    qc2_unitary = qiskit.quantum_info.Operator(qc2).data
    identity = np.eye(*qc2_unitary.shape)
    prod = np.conj(qc2_unitary).T @ qc1_unitary
    global_phase = prod[0][0]
    prod *= np.conj(global_phase)
    assert np.allclose(prod, identity)


def cnot_to_msgate(qc, check=False):
    qc_new = QuantumCircuit(*qc.qregs, *qc.cregs)
    for gate in qc.data:
        if gate.operation.name == "cx":
            control = gate.qubits[0]
            target = gate.qubits[1]
            qc_new.ry(np.pi / 2, control)
            qc_new.append(MSGate(0, 0, 1 / 4), [control, target])
            qc_new.rx(-np.pi / 2, control)
            qc_new.rx(-np.pi / 2, target)
            qc_new.ry(-np.pi / 2, control)
        else:
            qc_new.append(gate)
    if check:
        check_qc_compilation(qc_new, qc)
    return qc_new


def rx_ry_to_rz(qc, check=False):
    qc_new = QuantumCircuit(*qc.qregs, *qc.cregs)
    trivial_phases = np.array([0, 1 / 2, 1, 3 / 2]) * np.pi
    for gate in qc.data:
        if gate.operation.name == "rx":
            phase = gate.operation.params[0]
            if np.all(np.abs((trivial_phases - phase) % (2 * np.pi)) > 1e-7):
                qubit = gate.qubits[0]
                qc_new.ry(-np.pi / 2, qubit)
                qc_new.rz(phase, qubit)
                qc_new.ry(np.pi / 2, qubit)
            else:
                qc_new.append(gate)
        elif gate.operation.name == "ry":
            phase = gate.operation.params[0]
            if np.all(np.abs((trivial_phases - phase) % (2 * np.pi)) > 1e-7):
                qubit = gate.qubits[0]
                qc_new.rx(np.pi / 2, qubit)
                qc_new.rz(phase, qubit)
                qc_new.rx(-np.pi / 2, qubit)
            else:
                qc_new.append(gate)
        else:
            qc_new.append(gate)
    if check:
        check_qc_compilation(qc_new, qc)
    return qc_new


def rx_ry_trivial_phases_to_gpigates(qc, check=False):
    qc_new = QuantumCircuit(*qc.qregs, *qc.cregs)
    for gate in qc.data:
        if gate.operation.name == "rx":
            phase = gate.operation.params[0] % (2 * np.pi)
            qubit = gate.qubits[0]
            if abs(phase) < 1e-7 or abs(phase - (2 * np.pi)) < 1e-7:
                pass
            elif abs(phase - np.pi / 2) < 1e-7:
                qc_new.append(GPI2Gate(0), [qubit])
            elif abs(phase - np.pi) < 1e-7:
                qc_new.append(GPIGate(0), [qubit])
            elif abs(phase - 3 / 2 * np.pi) < 1e-7:
                qc_new.append(GPI2Gate(1 / 2), [qubit])
        elif gate.operation.name == "ry":
            phase = gate.operation.params[0] % (2 * np.pi)
            qubit = gate.qubits[0]
            if abs(phase) < 1e-7 or abs(phase - (2 * np.pi)) < 1e-7:
                pass
            elif abs(phase - np.pi / 2) < 1e-7:
                qc_new.append(GPI2Gate(1 / 4), [qubit])
            elif abs(phase - np.pi) < 1e-7:
                qc_new.append(GPIGate(1 / 4), [qubit])
            elif abs(phase - 3 / 2 * np.pi) < 1e-7:
                qc_new.append(GPI2Gate(3 / 4), [qubit])
        else:
            qc_new.append(gate)
    if check:
        check_qc_compilation(qc_new, qc)
    return qc_new


def consume_rz_gates(qc, check=False):
    qubit_phases = {}
    for qreg in qc.qregs:
        qubit_phases[qreg] = [0] * qreg.size
    
    qc_new = QuantumCircuit(*qc.qregs, *qc.cregs)
    for gate in qc.data:
        if gate.operation.name == "rz":
            gate_qubit = gate.qubits[0]
            qubit_phases[gate_qubit._register][gate_qubit._index] += gate.operation.params[0]
        elif gate.operation.name == "gpi":
            gate_qubit = gate.qubits[0]
            phase = (gate.operation.params[0] - qubit_phases[gate_qubit._register][gate_qubit._index] / (2 * np.pi)) % 1.0
            qc_new.append(GPIGate(phase), [gate_qubit])
        elif gate.operation.name == "gpi2":
            gate_qubit = gate.qubits[0]
            phase = (gate.operation.params[0] - qubit_phases[gate_qubit._register][gate_qubit._index] / (2 * np.pi)) % 1.0
            qc_new.append(GPI2Gate(phase), [gate_qubit])
        elif gate.operation.name == "ms":
            gate_qubits = [gate.qubits[0], gate.qubits[1]]
            assert gate.operation.params[0] == 0 and gate.operation.params[1] == 0
            assert gate.operation.params[2] == 1 / 4
            phase_0 = (-qubit_phases[gate_qubits[0]._register][gate_qubits[0]._index] / (2 * np.pi)) % 1.0
            phase_1 = (-qubit_phases[gate_qubits[1]._register][gate_qubits[1]._index] / (2 * np.pi)) % 1.0
            qc_new.append(MSGate(phase_0, phase_1, 1 / 4), gate_qubits)
        else:
            qc_new.append(gate)
    if check:
        qc_check = qc_new.copy()
        for i, phase in enumerate(qubit_phases):
            qc_check.rz(phase, i)
        check_qc_compilation(qc_check, qc)
    return qc_new