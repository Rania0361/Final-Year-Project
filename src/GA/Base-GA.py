"""
Abstract base class for Genetic Algorithms used in quantum circuit optimization.
This module contains all shared logic (representation, operators, evaluation)
that is inherited by both:
    - mono/single_objective_ga.py
    - multi/nsga2.py

Author: BOUHADOUZA Rania
Project: Quantum Circuit Design & Optimization via Genetic Algorithms
"""

import random
import numpy as np
from abc import ABC, abstractmethod
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator


# ──────────────────────────────────────────────
# Gate cost table (used for circuit cost metric)
# ──────────────────────────────────────────────
GATE_COSTS = {
    'NOT': 1, 'Phase': 1,
    'H': 2,
    'X': 1, 'Y': 1, 'Z': 1,
    'CX': 5,
    'SWAP': 11, 'Peres': 12, 'Toffoli': 13,
    'Fredkin': 19, 'Miller': 24,
    'RX': 1, 'RY': 1, 'RZ': 1,
}

# Gate metadata: does it need a control qubit or a rotation angle?
QUANTUM_GATES = {
    "H":  {"control": False, "rotation": False},
    "X":  {"control": False, "rotation": False},
    "Y":  {"control": False, "rotation": False},
    "Z":  {"control": False, "rotation": False},
    "CX": {"control": True,  "rotation": False},
    "RX": {"control": False, "rotation": True},
    "RY": {"control": False, "rotation": True},
    "RZ": {"control": False, "rotation": True},
}


# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
def set_random_seeds(seed: int = 42):
    """Set all random seeds for full reproducibility across runs."""
    np.random.seed(seed)
    random.seed(seed)
    print(f"[seed] All random seeds set to: {seed}")


# ──────────────────────────────────────────────
# Circuit representation
# gene = [target_qubit, gate_name, control_qubit, angle]
# ──────────────────────────────────────────────
def generate_gene(num_qubits: int) -> list:
    """
    Generate a single random gene (one quantum gate).

    A gene is encoded as:
        [target_qubit (int), gate_name (str), control_qubit (int|None), angle (float|None)]

    Parameters
    ----------
    num_qubits : int

    Returns
    -------
    list : [target, gate, control, angle]
    """
    target_qubit = np.random.randint(num_qubits)
    gate = np.random.choice(list(QUANTUM_GATES.keys()))
    control_qubit = None
    angle = None

    if QUANTUM_GATES[gate]["control"]:
        control_qubit = np.random.randint(num_qubits)
        while control_qubit == target_qubit:
            control_qubit = np.random.randint(num_qubits)

    if QUANTUM_GATES[gate]["rotation"]:
        angle = np.random.uniform(0, 2 * np.pi)

    return [target_qubit, gate, control_qubit, angle]


def generate_chromosome(num_qubits: int, num_genes: int) -> list:
    """Generate a chromosome as a sequence of genes."""
    return [generate_gene(num_qubits) for _ in range(num_genes)]


def generate_population(pop_size: int, num_qubits: int, num_genes: int) -> list:
    """Generate an initial population of chromosomes."""
    return [generate_chromosome(num_qubits, num_genes) for _ in range(pop_size)]


def create_quantum_circuit(chromosome: list, num_qubits: int) -> QuantumCircuit:
    """
    Decode a chromosome into a Qiskit QuantumCircuit.

    Parameters
    ----------
    chromosome : list of genes
    num_qubits : int

    Returns
    -------
    QuantumCircuit
    """
    circuit = QuantumCircuit(num_qubits)
    for gene in chromosome:
        target, gate, control, angle = gene
        if gate in ["H", "X", "Y", "Z"]:
            getattr(circuit, gate.lower())(target)
        elif gate == "CX":
            circuit.cx(control, target)
        elif gate in ["RX", "RY", "RZ"]:
            getattr(circuit, gate.lower())(angle, target)
    return circuit


# ──────────────────────────────────────────────
# Fitness metrics
# ──────────────────────────────────────────────
def compute_fidelity(circuit: QuantumCircuit, target_unitary: np.ndarray) -> float:
    """
    Two fidelity strategies are used in this project:

    [1] Exact Fidelity (this method) — Trace Overlap Formula:

            F(U, V) = |Tr(U_candidate · U_target†)| / 2^n

        where U_target† is the conjugate transpose of the target unitary
        and 2^n is the Hilbert space dimension (normalization factor).
        Used by: mono-objective GA, multi-objective GA (NSGA-II).

    [2] Approximate Fidelity (see fidelity/approx.py) — Local Evaluation + Aggregation:

        To avoid the exponential cost of full unitary simulation, the circuit
        is partitioned into sub-circuits. Exact fidelity is evaluated locally
        on each sub-circuit k against its corresponding sub-unitary:

            F_local(k) = |Tr(U_k · V_k†)| / 2^|k|

        The global fidelity is then approximated by aggregating (multiplying)
        all local fidelities:

            F_approx = ∏_k  F_local(k)

        Two partitioning strategies are supported:
            - Graph-based  (Louvain + Kernighan-Lin) : for sparse circuits
            - Sliding Window                          : for dense circuits
    """
# ──────────────────────────────────────────────
# Genetic operators (shared by mono & multi GA)
# ──────────────────────────────────────────────
def crossover(parent1: list, parent2: list) -> tuple:
    """
    Single-point crossover between two parent chromosomes.

    Returns
    -------
    (child1, child2) : tuple of two new chromosomes
    """
    min_length = min(len(parent1), len(parent2))
    if min_length < 2:
        return parent1.copy(), parent2.copy()

    cut_point = np.random.randint(1, min_length)
    child1 = parent1[:cut_point] + parent2[cut_point:]
    child2 = parent2[:cut_point] + parent1[cut_point:]
    return child1, child2


def mutate(chromosome: list, num_qubits: int, mutation_rate: float = 0.1) -> list:
    """
    Apply two types of mutation to a chromosome:
        1. Gene mutation     — replace a random gene with a new one
        2. Structural mutation — insert or delete a gene (prob=0.1 each)

    Parameters
    ----------
    chromosome    : list of genes (modified in place)
    num_qubits    : int
    mutation_rate : float, probability of gene mutation

    Returns
    -------
    list : mutated chromosome
    """
    # Gene-level mutation
    if np.random.rand() < mutation_rate and len(chromosome) > 0:
        idx = np.random.randint(len(chromosome))
        chromosome[idx] = generate_gene(num_qubits)

    # Structural mutation (insertion / deletion)
    if np.random.rand() < 0.1:
        if np.random.rand() < 0.5 and len(chromosome) > 1:
            del chromosome[np.random.randint(len(chromosome))]
        else:
            idx = np.random.randint(len(chromosome) + 1)
            chromosome.insert(idx, generate_gene(num_qubits))

    return chromosome


# ──────────────────────────────────────────────
# Abstract Base Class
# ──────────────────────────────────────────────
class BaseGA(ABC):
    """
    Abstract base class for all GA variants in this project.

    Subclasses must implement:
        - evaluate_individual()  : compute fitness for one chromosome
        - select()               : survivor / parent selection strategy
        - run()                  : main GA loop

    Shared methods (ready to use or override):
        - initialize_population()
        - crossover_op()
        - mutate_op()
        - decode()
    """

    def __init__(
        self,
        num_qubits: int,
        num_genes: int,
        pop_size: int,
        generations: int,
        target_unitary: np.ndarray,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.9,
    ):
        self.num_qubits = num_qubits
        self.num_genes = num_genes
        self.pop_size = pop_size
        self.generations = generations
        self.target_unitary = target_unitary
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self) -> list:
        """Generate the initial random population."""
        return generate_population(self.pop_size, self.num_qubits, self.num_genes)

    def decode(self, chromosome: list) -> QuantumCircuit:
        """Decode a chromosome into a QuantumCircuit."""
        return create_quantum_circuit(chromosome, self.num_qubits)

    def crossover_op(self, p1: list, p2: list) -> tuple:
        """Wrapper around the module-level crossover function."""
        return crossover(p1, p2)

    def mutate_op(self, chromosome: list) -> list:
        """Wrapper around the module-level mutate function."""
        return mutate(chromosome, self.num_qubits, self.mutation_rate)

    @abstractmethod
    def evaluate_individual(self, chromosome: list):
        """
        Compute and return the fitness of one chromosome.
        - Mono-objective GA  → returns a single float (fidelity)
        - Multi-objective GA → returns [fidelity, -depth, -cost]
        """
        ...

    @abstractmethod
    def select(self, population: list) -> list:
        """
        Select individuals to form the next generation.
        - Mono  → elitism / roulette / tournament
        - Multi → NSGA-II non-dominated sort + crowding distance
        """
        ...
    
    @abstractmethod
    def run(self):
        """
        Main GA loop. Returns the best solution(s) found.
        - Mono  → best individual (highest fidelity)
        - Multi → final Pareto front
        """
        ...
