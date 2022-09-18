import qiskit.providers.aer.noise as noise
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
        self.U = U
        self.bread_gates = bread_gates
        self.inner_backend = inner_backend

    def run_circuit_and_measure(self, circuit, n_samples):
        data_qubit_indices = tuple(range(circuit.n_qubits))
        new_circuit = Circuit([])
        n_sandwiches = 0

        for operation in circuit.operations:
            if operation.gate is self.U:
                for P in self.bread_gates:
                    n_sandwiches +=1
                    op_indices = operation.qubit_indices
                    control_qubit_index = circuit.n_qubits + n_sandwiches
                    controlled_P_qubits = (control_qubit_index,) + data_qubit_indices
                    print("U():")
                    print(type(self.U(*op_indices)))
                    print("P:")
                    print(type(P))
                    print("gate:")
                    print(type(self.U(*op_indices).gate))
                    print("Dagger:")
                    print(type(self.U(*op_indices).gate.dagger))
                    print("U.gate.dagger:")
                    try:
                        print(type(self.U.gate.dagger))
                    except:
                        pass
                    print("N qubits")
                    print(new_circuit.n_qubits)
                    print(type(new_circuit.n_qubits))
                    print("Lifted matrix:")
                    print(type(self.U(*op_indices).lifted_matrix(new_circuit.n_qubits)))
                    Pprime = self.U(*op_indices).lifted_matrix(new_circuit.n_qubits) * P * self.U(*op_indices).gate.dagger # make this run faster
                    new_circuit += Pprime.gate.controlled(1)(*controlled_P_qubits)
            else:
                new_circuit += operation

        raw_meas = self.inner_backend.run_circuit_and_measure(new_circuit, n_samples)

        raw_counts  = raw_meas.get_counts()
        sandwiched_counts = {}
        for key in raw_counts.keys():
            if "1" not in key[:circuit.n_qubits]:
                sandwiched_counts[key[:circuit.n_qubits]] = raw_counts[key]
        return Measurements.from_counts(sandwiched_counts)

# from orquestra.quantum.backends import PauliSandwichBackend
bread_gates = [X, Z]
sandwiched_qiskit_backend = PauliSandwichBackend(CNOT, bread_gates, qiskit_sim)
measurements = sandwiched_qiskit_backend.run_circuit_and_measure(circ, 1000)

# after sandwiching, we should have no errors
# getting the dictionary {"111", 1000} indicates errors have been eliminated
print(measurements.get_counts())
