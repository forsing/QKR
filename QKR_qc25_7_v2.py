import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="No gradient function provided")

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--no-plot", action="store_true")
_ARGS, _ = _ap.parse_known_args()

"""
QKR (Quantum Kernel Regressor)
"""


"""
https://github.com/forsing
https://github.com/forsing?tab=repositories
"""

"""
Loto Skraceni Sistemi
https://www.lotoss.info
ABBREVIATED LOTTO SYSTEMS
"""

"""
svih 4584 izvlacenja Loto 7/39 u Srbiji
30.07.1985.- 20.03.2026.
"""


import numpy as np
import pandas as pd
from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer

try:
    from qiskit_aer import AerSimulator
except ImportError:
    AerSimulator = None  # type: ignore[misc, assignment]

from qiskit.visualization import plot_histogram
from tqdm import tqdm

from qiskit_machine_learning.utils import algorithm_globals
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
algorithm_globals.random_seed = SEED

# =========================
# Učitaj CSV
# =========================
df = pd.read_csv("/data/loto7_4584_k23.csv", header=None).iloc[:, :7]
min_val = [1,2,3,4,5,6,7]
max_val = [33,34,35,36,37,38,39]

def map_to_indexed_range(df, min_val, max_val):
    df_indexed = df.copy()
    for i in range(df.shape[1]):
        df_indexed[i] = df[i] - min_val[i]
        if not df_indexed[i].between(0, max_val[i]-min_val[i]).all():
            raise ValueError(f"Kolona {i} izvan validnog opsega")
    return df_indexed

df_indexed = map_to_indexed_range(df, min_val, max_val)

# =========================
# QKR Parametri
# =========================
num_qubits = 5      # po poziciji
num_layers = 2
num_loto_cols = 7   # v2: 7 loto kolona — u simulaciji jedno kolo = samo 5 qubit-a (inače 7×5 → statevector ne staje u RAM)

if AerSimulator is not None:
    simulator = AerSimulator()
else:
    simulator = Aer.get_backend("qasm_simulator")
shots = 1024

def encode_position(value):
    v = int(value)
    bin_full = format(v, "b")
    # v2: ako vrednost treba više od num_qubits bitova, koristi poslednjih num_qubits (LSB)
    if len(bin_full) > num_qubits:
        bin_repr = bin_full[-num_qubits:]
    else:
        bin_repr = bin_full.zfill(num_qubits)
    qc = QuantumCircuit(num_qubits)
    for i, bit in enumerate(reversed(bin_repr)):
        if bit == "1":
            qc.x(i)
    return qc

def variational_layer(params):
    qc = QuantumCircuit(num_qubits)
    for i in range(num_qubits):
        qc.ry(params[i], i)
    for i in range(num_qubits-1):
        qc.cx(i, i+1)
    return qc

def qcbm_ansatz(params):
    qc = QuantumCircuit(num_qubits)
    for layer in range(num_layers):
        start = layer*num_qubits
        end = (layer+1)*num_qubits
        qc.compose(variational_layer(params[start:end]), inplace=True)
    return qc

def build_qcbm_circuit(params, value):
    """Jedna loto pozicija: encode(value) + QCBM (num_qubits) — usklađeno sa petljom predikcije."""
    qc = QuantumCircuit(num_qubits)
    qc.compose(encode_position(value), inplace=True)
    qc.compose(qcbm_ansatz(params), inplace=True)
    qc.measure_all()
    return qc

# =========================
# Predikcija QKR
# =========================
predicted_combination = []

last_value = df_indexed.iloc[-1].tolist()

for pos in range(num_loto_cols):
    print(f"\n--- QKR pozicija {pos+1} ---")
    # v2: deterministički parametri po poziciji (isti SEED + redni broj prolaza)
    _rng = np.random.default_rng(SEED + pos)
    params = _rng.uniform(0, 2 * np.pi, num_qubits * num_layers)

    # Circuit samo za trenutnu poziciju (5 qubit-a)
    qc = build_qcbm_circuit(params, last_value[pos])

    # Simuliraj qasm sa 1024 shots
    compiled_circuit = transpile(qc, simulator)
    result = simulator.run(compiled_circuit, shots=shots).result()
    counts = result.get_counts()
    
    # Najverovatniji izlaz
    most_probable = max(counts, key=counts.get)
    pred_val = int(most_probable[-num_qubits:],2)
    pred_val = max(min_val[pos], min(pred_val + min_val[pos], max_val[pos]))
    predicted_combination.append(pred_val)

print()
print("\n=== QKR Predviđena loto kombinacija (7) ===")
print(" ".join(str(x) for x in predicted_combination))

# opcionalno: histogram poslednje pozicije
if not _ARGS.no_plot:
    import matplotlib.pyplot as plt

    _out_dir = Path(__file__).resolve().parent / "QKR_qc25_7_v2_out"
    _out_dir.mkdir(parents=True, exist_ok=True)
    _hist_path = _out_dir / "histogram_last.png"
    _fig = plot_histogram(counts)
    if hasattr(_fig, "savefig"):
        _fig.savefig(_hist_path, dpi=150, bbox_inches="tight")
    else:
        plt.savefig(_hist_path, dpi=150, bbox_inches="tight")
        plt.close("all")
    print(f"[plot] Sačuvano: {_hist_path}")

print()



"""
python3 QKR_qc25_7_v2.py 
python3 QKR_qc25_7_v2.py --no-plot
"""


"""
--- QKR pozicija 1 ---

--- QKR pozicija 2 ---

--- QKR pozicija 3 ---

--- QKR pozicija 4 ---

--- QKR pozicija 5 ---

--- QKR pozicija 6 ---

--- QKR pozicija 7 ---


=== QKR Predviđena loto kombinacija (7) ===
12 x 18 30 y 15 z
"""
