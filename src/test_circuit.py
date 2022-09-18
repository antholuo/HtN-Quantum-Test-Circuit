import qiskit.providers.aer.noise as noise
from icecream import ic
from orquestra.integrations.qiskit.simulator import QiskitSimulator
from orquestra.quantum.circuits import CNOT, Circuit, X, Z

error = noise.depolarizing_error(0.1, 2)

# declare noise model
noise_model = noise.NoiseModel()
noise_model.add_all_qubit_quantum_error(error, ["cx"])

circ = Circuit([CNOT(0, 1), X(0), X(1), CNOT(1, 2)])

# declare backend and run circuit
qiskit_sim = QiskitSimulator(device_name="aer_simulator", noise_model=noise_model)
measurements = qiskit_sim.run_circuit_and_measure(circ, 1000)

# before sandwiching, should have some errors
print(measurements.get_counts())


class PauliSandwichBackend(QiskitSimulator):
    def __init__(self, U, bread_gates, inner_backend):
        print("init", U, bread_gates, inner_backend)
        # U is going to be a gate operation (at this point, CNOT)
        self.U = U

        # bread gates is going to be a bunch of Circuits/gates (not super sure)
        self.bread_gates = bread_gates
        self.inner_backend = inner_backend

    def run_circuit_and_measure(self, circuit, n_samples):
        data_qubit_indices = tuple(range(circuit.n_qubits))
        new_circuit = Circuit([])
        n_sandwiches = 0

        for operation in circuit.operations:
            if operation.gate is self.U:
                for P in self.bread_gates:
                    ic(operation)
                    ic(P)
                    n_sandwiches +=1
                    op_indices = operation.qubit_indices
                    control_qubit_index = circuit.n_qubits + n_sandwiches
                    controlled_P_qubits = (control_qubit_index,) + data_qubit_indices
                    Pprime = self.U(*op_indices).gate * P * self.U.gate.dagger(*op_indices) # make this run faster
                    Pprime = P;
                    
                    new_circuit += Pprime.gate.controlled(1)(*controlled_P_qubits)
                    new_circuit += operation
                    new_circuit += P.gate.controlled(1)(*controlled_P_qubits)
            else:
                new_circuit += operation

        # runs on their inner backend our new circuit (i.e. our run_circuit_and_measure is a pre-processor)
        raw_meas = self.inner_backend.run_circuit_and_measure(new_circuit, n_samples)

        raw_counts  = raw_meas.get_counts()
        sandwiched_counts = {}
        for key in raw_counts.keys():
            if "1" not in key[:circuit.n_qubits]:
                sandwiched_counts[key[:circuit.n_qubits]] = raw_counts[key]
        return measurements.from_counts(sandwiched_counts)

# from orquestra.quantum.backends import PauliSandwichBackend

# sandwiched_qiskit_backend = PauliSandwichBackend(CNOT, None, qiskit_sim)
pauli_bread_gates = [X, Z]
sandwiched_qiskit_backend = PauliSandwichBackend(CNOT, pauli_bread_gates, qiskit_sim)
measurements = sandwiched_qiskit_backend.run_circuit_and_measure(circ, 1000)

# after sandwiching, we should have no errors
# getting the dictionary {"111", 1000} indicates errors have been eliminated
print(measurements.get_counts())
