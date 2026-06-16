import os
import json
import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import expm_multiply
from tqdm import tqdm

# ── QuSpin imports ────────────────────────────────────────────────────────────
from quspin.operators import hamiltonian
from quspin.basis import spin_basis_1d
from quspin.tools.measurements import ent_entropy

# ── Parameters ────────────────────────────────────────────────────────────────
J       = 1.0          # exchange coupling (energy unit)
DELTA   = 1.0          # XXZ anisotropy (Heisenberg point)

N_SIZES = [4, 6, 8, 10]

W_VALUES = [0.5, 3.5, 8.0]

N_DIS = {4: 1, 6: 1, 8: 1, 10: 1}
T_MAX   = 100.0        # max time  [J^{-1}]
DT      = 0.05          # time step [J^{-1}]
T_SAT   = 10.0         # discard transient before this for time-averaged S_ent

OUTDIR  = "./data"
SEED    = 42           # base random seed

# ── Helpers ───────────────────────────────────────────────────────────────────

def make_basis(N):
    """Spin-1/2 basis in the Sz_tot=0 sector."""
    return spin_basis_1d(N, Nup=N // 2)

def make_tfim_basis(N):
    """
    Spin-1/2 basis for TFIM.
    We must use the full Hilbert space because the transverse field (S^x) 
    does not conserve total S^z magnetization.
    """
    return spin_basis_1d(N)


def make_hamiltonian(N, basis, disorder):
    """
    Build QuSpin Hamiltonian for one disorder realization.

    disorder : array of shape (N,) — on-site fields h_i
    """
    # Hopping / exchange terms  (open boundary, i=0..N-2)
    hop = [[J * 0.5, i, i + 1] for i in range(N - 1)]   # S^+_i S^-_{i+1} + h.c.
    zz  = [[J * DELTA, i, i + 1] for i in range(N - 1)]  # S^z_i S^z_{i+1}

    # On-site disorder
    hz  = [[disorder[i], i] for i in range(N)]

    static = [
        ["+-", hop],
        ["-+", hop],
        ["zz", zz],
        ["z",  hz],
    ]
    dynamic = []

    H = hamiltonian(
        static, dynamic,
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False,
    )
    return H


def make_tfim_hamiltonian(N, basis, J_ising, h_x):
    """
    Build QuSpin Hamiltonian for the Transverse-Field Ising Model (OBC).
    
    J_ising : float — Coupling strength for S^z_i S^z_{i+1}
    h_x     : float — Transverse magnetic field strength for S^x_i
    """
    # Ising interaction term (open boundary, i=0..N-2)
    zz = [[J_ising, i, i + 1] for i in range(N - 1)]
    
    # Transverse field term
    x_field = [[h_x, i] for i in range(N)]

    static = [
        ["zz", zz],
        ["x", x_field],
    ]
    dynamic = []

    H = hamiltonian(
        static, dynamic,
        basis=basis,
        dtype=np.float64,
        check_symm=False,
        check_herm=False,
        check_pcon=False, # We disabled particle conservation since Nup is removed
    )
    return H

def neel_state(N, basis):
    """
    Return the Neel state |up down up down ...> as a dense vector
    in the given basis.
    """
    # QuSpin integer representation: spin config as binary string
    # up=1, down=0; site 0 is leftmost
    # Neel: sites 0,2,4,... are up (1); sites 1,3,5,... are down (0)
    neel_int = sum(1 << (N - 1 - i) for i in range(0, N, 2))
    state = np.zeros(basis.Ns, dtype=np.complex128)
    idx = basis.index(neel_int)
    state[idx] = 1.0
    return state


def time_evolve(H, psi0, times, N):
    """
    Time-evolve psi0 under H and return states at each time in `times`.
    Uses expm_multiply (Krylov) for large N, dense expm for small N.
    """
    dt    = times[1] - times[0]
    n_t   = len(times)
    psi   = psi0.astype(np.complex128).copy()
    Hmat  = H.tocsr()                # sparse CSR matrix

    states = np.zeros((n_t, len(psi0)), dtype=np.complex128)
    states[0] = psi

    # Propagator for one step: exp(-i H dt)
    # expm_multiply applies exp(A) @ v efficiently
    for t_idx in range(1, n_t):
        psi = expm_multiply(-1j * dt * Hmat, psi)
        states[t_idx] = psi

    return states


def compute_sz_all(states, N, basis):
    """
    Compute <Sz_i(t)> for all sites i.
    Returns array of shape (N, T).
    """
    sz_all = np.zeros((N, states.shape[0]), dtype=np.float64)
    for i in range(N):
        op_list = [["z", [[1.0, i]]]]
        Sz_i = hamiltonian(
            op_list, [],
            basis=basis,
            dtype=np.float64,
            check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        for t_idx, psi in enumerate(states):
            sz_all[i, t_idx] = Sz_i.expt_value(psi).real
    return sz_all


def compute_sent(states, N, basis):
    """
    Compute half-chain entanglement entropy S_ent(t).
    Subsystem A = sites 0 .. N//2-1.
    Returns array of shape (T,).
    """
    sub_sys_A = list(range(N // 2))
    sent = np.zeros(states.shape[0], dtype=np.float64)
    for t_idx, psi in enumerate(states):
        ent_dict = ent_entropy(
            psi, basis,
            chain_subsys=sub_sys_A,
            return_rdm=None,
            alpha=1,          # von Neumann entropy
        )
        sent[t_idx] = ent_dict["Sent_A"]
    return sent


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    times = np.arange(0.0, T_MAX + DT, DT)
    np.save(os.path.join(OUTDIR, "tgrid.npy"), times)
    n_t = len(times)

    meta = {
        "J": J, "Delta": DELTA,
        "N_sizes": N_SIZES,
        "W_values": W_VALUES,
        "N_dis": N_DIS,
        "T_max": T_MAX, "dt": DT, "T_sat": T_SAT,
        "seed": SEED,
    }
    with open(os.path.join(OUTDIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta.json and tgrid.npy")

    rng = np.random.default_rng(SEED)

    for N in N_SIZES:
        print(f"\n{'='*60}")
        print(f"  N = {N}  (Hilbert space dim in Sz=0 sector)")
        print(f"{'='*60}")

        # basis = make_basis(N)
        # print(f"  Basis size: {basis.Ns}")

        # psi0 = neel_state(N, basis)
        # center = N // 2

        # for W in W_VALUES:
        #     n_real = N_DIS[N]
        #     print(f"\n  W = {W:.1f},  n_realizations = {n_real}")

        #     for r in tqdm(range(n_real), desc=f"  N={N} W={W}"):
        #         # ── disorder realization ──────────────────────────────────
        #         h_fields = rng.uniform(-W, W, size=N)
        #         H = make_hamiltonian(N, basis, h_fields)

        #         # ── time evolution ────────────────────────────────────────
        #         states = time_evolve(H, psi0, times, N)

        basis = make_tfim_basis(N)
        psi0 = neel_state(N, basis)
        center = N // 2
        
        J_ising = 1.0
        HX_VALUES = [0.1, 1.0, 8.0] # Replace W_VALUES with transverse fields

        for h_x in HX_VALUES:
            n_real = N_DIS[N]
            print(f"\n  h_x = {h_x:.1f},  n_realizations = {n_real}")
            for r in tqdm(range(n_real), desc=f"  N={N} h_x={h_x}"):
                H = make_tfim_hamiltonian(N, basis, J_ising, h_x)
                states = time_evolve(H, psi0, times, N)
                # ── center-site <Sz(t)> ───────────────────────────────────
                sz_center = np.zeros(n_t, dtype=np.float64)
                op_center = hamiltonian(
                    [["z", [[1.0, center]]]], [],
                    basis=basis, dtype=np.float64,
                    check_symm=False, check_herm=False, check_pcon=False,
                )
                for t_idx, psi in enumerate(states):
                    sz_center[t_idx] = op_center.expt_value(psi).real

                fname_sz = f"TFIM_sz_dynamics_N{N}_h_x{h_x:.1f}_real{r}.npy"
                np.save(os.path.join(OUTDIR, fname_sz), sz_center)

                # ── all-site <Sz_i(t)> ────────────────────────────────────
                sz_all = compute_sz_all(states, N, basis)
                fname_szall = f"TFIM_sz_all_N{N}_h_x{h_x:.1f}_real{r}.npy"
                np.save(os.path.join(OUTDIR, fname_szall), sz_all)

                # ── entanglement entropy ──────────────────────────────────
                sent = compute_sent(states, N, basis)
                fname_sent = f"TFIM_sent_N{N}_h_x{h_x:.1f}_real{r}.npy"
                np.save(os.path.join(OUTDIR, fname_sent), sent)

    print("\nDone. All files saved to", OUTDIR)


if __name__ == "__main__":
    main()