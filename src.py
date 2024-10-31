from abc import ABC, abstractmethod
from collections import defaultdict
from types import NotImplementedType

import numpy as np
from qiskit import QuantumCircuit, generate_preset_pass_manager
from qiskit.circuit import Parameter
from qiskit.result import marginal_counts
from qiskit_aer import AerSimulator
from sklearn.metrics import log_loss


def output_mapping(binary_counts, n_classes):
    """
    Maps binary count output to n classes
    """
    mapping = {}
    class_preds = defaultdict(int)
    for binary in binary_counts:
        b_number = int(binary, base=2)
        c = b_number % n_classes
        mapping.update({binary: c})

    for binary, count in binary_counts.items():
        class_preds[mapping[binary]] += count

    for i in range(n_classes):
        if i not in class_preds:
            class_preds[i] = 0

    return class_preds


BACKEND = AerSimulator()


def measure_circuit(qc: QuantumCircuit, shots: int = 1000, qbits: list | None = None):
    pm = generate_preset_pass_manager(backend=BACKEND, optimization_level=1)
    isa_circuit = pm.run(qc)
    result = BACKEND.run(isa_circuit, shots=shots).result()
    if qbits is not None:
        return marginal_counts(result, qbits).get_counts()
    else:
        return result.get_counts()


# Circuits
def Custom_UnitaryGate1(parameters):
    qc = QuantumCircuit(2)
    qc.cx(0,1)
    qc.rz(parameters[0], 1)
    qc.ry(parameters[1], 0)
    qc.cx(0,1)
    return qc.to_gate(label="U1")

def Custom_UnitaryGate2(parameters):
    qc = QuantumCircuit(2)
    qc.cx(0,1)
    qc.rx(parameters[0], 1)
    qc.rx(parameters[1], 0)
    qc.rz(parameters[2], 0)
    qc.rz(parameters[3], 1)
    qc.cx(0,1)
    return qc.to_gate(label="U2")

def Custom_UnitaryGateV(parameters):
    qc = QuantumCircuit(1)
    qc.rx(parameters[0], 0)
    qc.ry(parameters[1], 0)
    qc.rz(parameters[2], 0)
    return qc.to_gate(label="V")

def circuit1(features, parameters):
    qc_1 = QuantumCircuit(4, 4)
    for i in range(4):
        qc_1.rx(qubit=i, theta=features[i])
    qc_1.barrier()

    #qc_1_unitary_V2 = UnitaryGate([[1,0], [0,1]])
    params = [Parameter(f"{i}") for i in range(9)]
    qc_1.append(Custom_UnitaryGate1([params[0], params[1]]), [3,2])
    qc_1.append(Custom_UnitaryGate1([params[0], params[1]]), [1,0])
    qc_1.barrier()

    qc_1.measure(qubit=2, cbit=2)
    qc_1.append(Custom_UnitaryGateV([params[2], params[3], params[4]]).control(1), [2,3])

    qc_1.measure(qubit=0, cbit=0)
    qc_1.append(Custom_UnitaryGateV([params[2], params[3], params[4]]).control(1), [0,1])
    
    qc_1.barrier()
    qc_1.append(Custom_UnitaryGate2([params[5], params[6], params[7], params[8]]), [3,1])
    qc_1.measure(qubit=1, cbit=1)
    qc_1.measure(qubit=3, cbit=3)

    qc_1 = qc_1.assign_parameters(parameters)
    return qc_1



def circuit2(features, trainable_parameters, layers=2):
    """
    Circuit implementing real amplitudes
    """
    input_size = len(features)
    qc = QuantumCircuit(input_size)
    for i in range(input_size):
        qc.rx(features[i], i)
    qc.barrier()

    for i in range(layers):
        for j in range(input_size):
            qc.ry(Parameter(f"phi{i}{j}"), j)
        for j in range(input_size - 1, 0, -1):
            qc.cx(j, j - 1)
        qc.barrier()

    qc = qc.assign_parameters(trainable_parameters)
    qc.measure_all()
    return qc


class BaseModel(ABC):
    """
    Base model for QNNs
    """
    @abstractmethod
    def __init__(
        self,
        learning_rate=0.01,
        prediction_shots=1000,
        gradient_shots=100,
        epsilon=1,
    ):
        self.learning_rate = learning_rate
        self.gradient_shots = gradient_shots
        self.prediction_shots = prediction_shots
        self.train_loss = []
        self.val_loss = []
        self.epsilon = epsilon

        self.circuit_params = []
        self.parameters: np.ndarray = NotImplementedType
        self.circuit_func = NotImplementedType

    def gradient(self, data, targets):
        gradients = []
        for i in range(len(self.parameters)):
            p1 = self.parameters.copy()
            p1[i] += self.epsilon
            p2 = self.parameters.copy()
            p2[i] -= self.epsilon
            loss1 = self._loss(data, targets, self.gradient_shots, p1)
            loss2 = self._loss(data, targets, self.gradient_shots, p2)
            partial = (loss1 - loss2) / 2 * self.epsilon
            gradients.append(partial)
        return np.array(gradients)

    def _loss(self, data, targets, shots, parameters=None):
        if parameters is None:
            parameters = self.parameters
        preds = np.array(
            [self._predict_proba(x, shots=shots, parameters=parameters) for x in data]
        )
        return log_loss(targets, preds, labels=[0, 1, 2])

    def fit(self, data, targets, val_data=None, val_targets=None):
        for i in range(5):
            print(f"Epoch {i+1}", end=" ")
            gradient = self.gradient(data, targets)
            self.parameters -= self.learning_rate * gradient

            train_loss = self._loss(data, targets, shots=self.prediction_shots)
            print(f"Train loss: {train_loss}", end=" ")
            self.train_loss.append(train_loss)
            if val_data is not None and val_targets is not None:
                val_loss = self._loss(
                    val_data, val_targets, shots=self.prediction_shots
                )
                self.val_loss.append(val_loss)
                print(f"Validation loss: {val_loss}")
        return self

    def _predict(self, x, shots=1000):
        circuit = self.circuit_func(x, self.parameters, *self.circuit_params)
        results = measure_circuit(circuit, shots)
        class_output = output_mapping(results, 3)
        prediction = max(class_output, key=lambda x: class_output[x])
        return prediction

    def predict(self, X):
        return np.array([self._predict(x, shots=self.prediction_shots) for x in X])

    def _predict_proba(self, x, shots=100, parameters=None):
        if parameters is None:
            parameters = self.parameters

        circuit = self.circuit_func(x, parameters, *self.circuit_params)
        results = measure_circuit(circuit, shots)
        class_output = sorted(output_mapping(results, 3).items())
        probs = np.array([n / shots for _, n in class_output])
        return probs

    def predict_proba(self, X):
        return np.array(
            [self._predict_proba(x, shots=self.prediction_shots) for x in X]
        )


class Model2(BaseModel):
    """
    Model for real amplitudes
    """
    def __init__(
        self,
        layers=2,
        learning_rate=0.01,
        prediction_shots=1000,
        gradient_shots=100,
        epsilon=1,
    ):
        super().__init__(learning_rate, prediction_shots, gradient_shots, epsilon)
        self.parameters = np.random.uniform(low=0, high=np.pi, size=(layers * 4,))
        self.circuit_func = circuit2
        self.circuit_params = [layers]

class Model1(BaseModel):
    """
    Model for convolutional neral network
    """
    def __init__(
        self,
        learning_rate=0.01,
        prediction_shots=1000,
        gradient_shots=100,
        epsilon=1,
    ):
        super().__init__(learning_rate, prediction_shots, gradient_shots, epsilon)
        self.parameters = np.random.uniform(low=0, high=np.pi, size=9)
        self.circuit_func = circuit1
