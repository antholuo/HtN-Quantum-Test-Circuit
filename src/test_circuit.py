import qiskit.providers.aer.noise as noise
from orquestra.integrations.qiskit.simulator import QiskitSimulator
from orquestra.quantum.circuits import CNOT, Circuit, X, Z, XX, ZZ, MatrixFactoryGate, CustomGateDefinition, ControlledGate
import qiskit

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
                    # print("operation:")
                    # print(operation)
                    # print(type(operation))
                    op_indices = operation.qubit_indices
                    control_qubit_index = circuit.n_qubits + n_sandwiches
                    controlled_P_qubits = (control_qubit_index,) + data_qubit_indices
                    # print("U:")
                    # print(type(self.U))
                    # print("U:")
                    # print(type(self.U))
                    # print("U():")
                    # print(type(self.U(*op_indices)))
                    # print("P:")
                    # print(type(P))
                    # print("gate:")
                    # print(type(self.U(*op_indices).gate))
                    # print("Dagger:")
                    # print(type(self.U(*op_indices).gate.dagger))
                    # print("U.gate.dagger:")
                    # print("N qubits")
                    # print(new_circuit.n_qubits)
                    # print(type(new_circuit.n_qubits))

                    # _U = self.U(*op_indices).gate
                    _U = self.U(*op_indices).gate
                    _U_prime_matrix = _U.matrix.inv()
                    print("U:")
                    print(_U.matrix)
                    print(type(_U))
                    print("P:")
                    print(P.matrix)
                    print(type(P))
                    print("U':")
                    print(_U_prime_matrix)
                    print(type(_U_prime_matrix))
                    Pprime_matrix =  _U.matrix * P.matrix *  _U_prime_matrix # make this run faster
                    print("P:")
                    print(Pprime_matrix)
                    Pprime = CustomGateDefinition("P'", Pprime_matrix, ())()
                    pprime_controlled_op = Pprime.controlled(1)(*controlled_P_qubits)
                    # print(type(pprime_controlled_op.gate))
                    # print(type(pprime_controlled_op.matrix_factory))
                    new_circuit += pprime_controlled_op
                    new_circuit += operation
                    new_circuit += P.controlled(1)(*controlled_P_qubits)

            else:
                new_circuit += operation
        operation = new_circuit.operations[0]
        gate_op = operation
        q_circuit = qiskit.QuantumCircuit(circuit.n_qubits)
        def qiskit_qubit(index: int, num_qubits_in_circuit: int) -> qiskit.circuit.Qubit:
            return qiskit.circuit.Qubit(
                qiskit.circuit.QuantumRegister(num_qubits_in_circuit, "q"), index
            )
        def _export_controlled_gate(gate, applied_qubit_indices, n_qubits_in_circuit, custom_names):
            if not isinstance(gate, ControlledGate):
                # Raising an exception here is redundant to the type hint, but it allows us
                # to handle exporting all gates in the same way, regardless of type
                raise ValueError(f"Can't export gate {gate} as a controlled gate")

            target_indices = applied_qubit_indices[gate.num_control_qubits :]
            target_gate, _, _ = _export_gate_to_qiskit(
                gate.wrapped_gate,
                applied_qubit_indices=target_indices,
                n_qubits_in_circuit=n_qubits_in_circuit,
                custom_names=custom_names,
            )
            controlled_gate = target_gate.control(gate.num_control_qubits)
            qiskit_qubits = [
                qiskit_qubit(qubit_i, n_qubits_in_circuit) for qubit_i in applied_qubit_indices
            ]
            return controlled_gate, qiskit_qubits, []

        def _export_custom_gate(gate, applied_qubit_indices, n_qubits_in_circuit, custom_names):
            if gate.name not in custom_names:
                raise ValueError(
                    f"Can't export gate {gate} as a custom gate, the circuit is missing its "
                    "definition"
                )

            if gate.params:
                raise ValueError(
                    f"Can't export parametrized gate {gate}, Qiskit doesn't support "
                    "parametrized custom gates"
                )
            # At that time of writing it Qiskit doesn't support parametrized gates defined with
            # a symbolic matrix.
            # See https://github.com/Qiskit/qiskit-terra/issues/4751 for more info.

            qiskit_qubits = [
                qiskit_qubit(qubit_i, n_qubits_in_circuit) for qubit_i in applied_qubit_indices
            ]
            qiskit_matrix = np.array(gate.matrix)
            return (
                qiskit.extensions.UnitaryGate(qiskit_matrix, label=gate.name),
                qiskit_qubits,
                [],
            )


        def _export_gate_to_qiskit(gate, applied_qubit_indices, n_qubits_in_circuit, custom_names):
            try:
                print("EXPORTED CONTROLLED GATE")
                return _export_controlled_gate(
                    gate, applied_qubit_indices, n_qubits_in_circuit, custom_names)
            except ValueError:
                pass

            print("EXPORTED CUSTOM GATE")
            return _export_custom_gate(
                gate, applied_qubit_indices, n_qubits_in_circuit, custom_names
            )

        _export_gate_to_qiskit(
            gate_op.gate,
            applied_qubit_indices=gate_op.qubit_indices,
            n_qubits_in_circuit=circuit.n_qubits,
            custom_names={},
        )
        print(operation)
        print(operation.gate.wrapped_gate)
        print(type(operation.gate.wrapped_gate))
        print(operation.gate.wrapped_gate.matrix_factory)
        print(type(operation.gate.wrapped_gate.matrix_factory))

        print(new_circuit.collect_custom_gate_definitions())
        raw_meas = self.inner_backend.run_circuit_and_measure(new_circuit, n_samples)

        raw_counts  = raw_meas.get_counts()
        sandwiched_counts = {}
        for key in raw_counts.keys():
            if "1" not in key[:circuit.n_qubits]:
                sandwiched_counts[key[:circuit.n_qubits]] = raw_counts[key]
        return Measurements.from_counts(sandwiched_counts)

# from orquestra.quantum.backends import PauliSandwichBackend
bread_gates = [CNOT]
sandwiched_qiskit_backend = PauliSandwichBackend(CNOT, bread_gates, qiskit_sim)
measurements = sandwiched_qiskit_backend.run_circuit_and_measure(circ, 1000)

# after sandwiching, we should have no errors
# getting the dictionary {"111", 1000} indicates errors have been eliminated
print(measurements.get_counts())
