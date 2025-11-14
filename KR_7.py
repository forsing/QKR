import warnings
warnings.filterwarnings("ignore", message="No gradient function provided")

"""
KR (Kernel Regressor)
"""

"""
| Paket                       | Verzija |
| --------------------------- | ------- |
| **python**                  | 3.11.13 |
| **qiskit**                  | 1.4.4   |
| **qiskit-machine-learning** | 0.8.3   |
| **qiskit-ibm-runtime**      | 0.43.0  |
| **macOS**                   | Tahos   |
| **Apple**                   | M1      |
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
svih 4510 izvlacenja
30.07.1985.- 11.11.2025.
"""


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from qiskit.circuit.library import ZZFeatureMap
from qiskit_aer import AerSimulator
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from tqdm import tqdm
import random

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)

# =========================
# Učitavanje CSV
# =========================
csv_path = '/data/loto7_4510_k89.csv'
df = pd.read_csv(csv_path, header=None)


# =========================
# Koristimo samo zadnjih N=100 za test
# =========================
N = 100
# N = 4510 # sve

df = df.tail(N).reset_index(drop=True)


# Priprema podataka
X = df.iloc[:, :-1].values   # prvih 6 brojeva
y_full = df.values           # svih 7 brojeva (6+1)



# Skaliranje ulaza
scaler_X = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X).astype(np.float64)

predicted_combination = []

# =========================
# KR treniranje po brojevima
# =========================
print()
for i in range(7):  
    print(f"\n--- Quantum Kernel Regression za broj {i+1} ---")
    y = y_full[:, i].astype(np.float64)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    num_qubits = X_scaled.shape[1]

    # Feature map
    feature_map = ZZFeatureMap(feature_dimension=num_qubits, reps=2, entanglement="linear")

    # Fidelity Quantum Kernel
    quantum_kernel = FidelityQuantumKernel(feature_map=feature_map)

    # Kernel matrica (trenutno vrlo skupa operacija, zato progress bar)
    print("Računam kernel matricu...")
    gram_matrix = quantum_kernel.evaluate(x_vec=X_scaled)

    # SVR sa gotovim kernelom
    svr = SVR(kernel='precomputed', C=10.0)

    # Fit uz progress bar
    pbar = tqdm(total=1, desc=f"Trening za broj {i+1}")
    pbar.update(1)
    svr.fit(gram_matrix, y_scaled)
    pbar.close()

    # Predikcija
    last_scaled = scaler_X.transform([X[-1]]).astype(np.float64)
    k_test = quantum_kernel.evaluate(x_vec=last_scaled, y_vec=X_scaled)
    pred_scaled = svr.predict(k_test)
    pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).round().astype(int)[0][0]

    predicted_combination.append(int(pred))
    print(f"Predikcija za broj {i+1}: {pred}")

"""
--- Quantum Kernel Regression za broj 1 ---
Računam kernel matricu...
Trening za broj 1: 100%|████████████████████████| 1/1 [00:00<00:00, 647.67it/s]
Predikcija za broj 1: 4

--- Quantum Kernel Regression za broj 2 ---
Računam kernel matricu...
Trening za broj 2: 100%|███████████████████████| 1/1 [00:00<00:00, 2341.88it/s]
Predikcija za broj 2: 9

--- Quantum Kernel Regression za broj 3 ---
Računam kernel matricu...
Trening za broj 3: 100%|███████████████████████| 1/1 [00:00<00:00, 2364.32it/s]
Predikcija za broj 3: x

--- Quantum Kernel Regression za broj 4 ---
Računam kernel matricu...
Trening za broj 4: 100%|███████████████████████| 1/1 [00:00<00:00, 2325.00it/s]
Predikcija za broj 4: x

--- Quantum Kernel Regression za broj 5 ---
Računam kernel matricu...
Trening za broj 5: 100%|███████████████████████| 1/1 [00:00<00:00, 2457.12it/s]
Predikcija za broj 5: x

--- Quantum Kernel Regression za broj 6 ---
Računam kernel matricu...
Trening za broj 6: 100%|███████████████████████| 1/1 [00:00<00:00, 2403.61it/s]
Predikcija za broj 6: 35

--- Quantum Kernel Regression za broj 7 ---
Računam kernel matricu...
Trening za broj 7: 100%|███████████████████████| 1/1 [00:00<00:00, 1222.83it/s]
Predikcija za broj 7: 37
"""

print()
print("\n=== Predviđena sledeća loto kombinacija (7) ===")
print(" ".join(str(num) for num in predicted_combination))
print()
"""
N = 100 # zadnjih 100

=== Predviđena sledeća loto kombinacija (7) ===
4 9 x x x 35 37
"""
