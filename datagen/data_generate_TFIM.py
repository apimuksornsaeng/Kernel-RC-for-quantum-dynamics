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

N_SIZES = [4,6,8,10,12]

HX_VALUES = [0.8,1.0,8.0]

N_DIS = 1               # number of realizations per N, h_x (set to 1 for TFIM since no disorder)
T_MAX   = 1500.0        # max time  [J^{-1}]
DT      = 0.05         # time step [J^{-1}]

OUTDIR  = "./data"
SYSTEM  = "TFIM"
STATE  = "Neel"       # initial state: "Neel" or "Paramagnet"
SEED    = 42           # base random seed
BC      = "periodic"  # 'open' or 'periodic'
if BC == 'periodic':
    SYSTEM += f'_{BC}'
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


def make_tfim_hamiltonian(N, basis, J_ising, h_x, BC = 'open'):
    """
    Build QuSpin Hamiltonian for the Transverse-Field Ising Model (OBC).
    
    J_ising : float — Coupling strength for S^z_i S^z_{i+1}
    h_x     : float — Transverse magnetic field strength for S^x_i
    """
    # Ising interaction term (open boundary, i=0..N-2)
    if BC == 'open':
        zz = [[J_ising, i, (i + 1)] for i in range(N - 1)]
    elif BC == 'periodic':
        zz = [[J_ising, i, (i + 1)%N] for i in range(N)]
    
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

def paramagnet_state(N, basis):
    """
    Return the paramagnetic state |+ + + ...> (all spins in x-direction)
    as a dense vector in the given basis.
    """
    # In Sz basis, |+> = (|up> + |down>) / sqrt(2)
    # For N spins, the state is a superposition of all 2^N configurations
    # The paramagnetic state is an equal superposition of all 2^N configurations
    state = np.ones(basis.Ns, dtype=np.complex128) / np.sqrt(basis.Ns)
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

def compute_szsz_all(states, N, basis):
    """
    Compute <Sz_i(t) Sz_{N/2}(t)> for all i != N/2.
    Returns array of shape (N-1, T).
    """
    szsz_all = np.zeros((N, states.shape[0]), dtype=np.float64)
    for i in range(N):
        if i != N // 2:
            op_list = [["zz", [[1.0, i, N//2]]]]
            SzSz_ij = hamiltonian(
                op_list, [],
                basis=basis,
                dtype=np.float64,
                check_symm=False,
            check_herm=False,
            check_pcon=False,
        )
        else:
            continue
        for t_idx, psi in enumerate(states):
            szsz_all[i, t_idx] = SzSz_ij.expt_value(psi).real
    return szsz_all


# ── Main loop ─────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTDIR, exist_ok=True)

    times = np.arange(0.0, T_MAX + DT, DT)
    np.save(os.path.join(OUTDIR, f"{SYSTEM}_tgrid.npy"), times)
    n_t = len(times)

    meta = {
        "J": J,
        "N_sizes": N_SIZES,
        "HX_values": HX_VALUES,
        "STATE": STATE,
        "N_dis": N_DIS,
        "T_max": T_MAX, "dt": DT,
        "seed": SEED,
    }
    with open(os.path.join(OUTDIR, f"{SYSTEM}_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved meta.json and tgrid.npy")

    rng = np.random.default_rng(SEED)

    for N in N_SIZES:
        print(f"\n{'='*60}")
        print(f"  N = {N} spins")
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
        if STATE == "Neel":
            psi0 = neel_state(N, basis)
        elif STATE == "Paramagnet":
            psi0 = paramagnet_state(N, basis)
        center = N // 2
        
        

        for h_x in HX_VALUES:
            n_real = N_DIS
            print(f"\n  h_x = {h_x:.1f},  n_realizations = {n_real}")
            for r in tqdm(range(n_real), desc=f"  N={N} h_x={h_x}"):
                H = make_tfim_hamiltonian(N, basis, J, h_x, BC)
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

                fname_sz = f"{SYSTEM}_sz_dynamics_N{N}_hx{h_x:.1f}_{STATE}_{r}.npy"
                np.save(os.path.join(OUTDIR, fname_sz), sz_center)

                # ── all-site <Sz_i(t)> ────────────────────────────────────
                sz_all = compute_sz_all(states, N, basis)
                fname_szall = f"{SYSTEM}_sz_all_N{N}_hx{h_x:.1f}_{STATE}_{r}.npy"
                np.save(os.path.join(OUTDIR, fname_szall), sz_all)

                # ── all-site <Sz_i(t) Sz_{N/2}(t)> ───────────────────────────────
                szsz_all = compute_szsz_all(states, N, basis)
                fname_szsz = f"{SYSTEM}_szsz_all_N{N}_hx{h_x:.1f}_{STATE}_{r}.npy"
                np.save(os.path.join(OUTDIR, fname_szsz), szsz_all)

                # ── entanglement entropy ──────────────────────────────────
                sent = compute_sent(states, N, basis)
                fname_sent = f"{SYSTEM}_sent_N{N}_hx{h_x:.1f}_{STATE}_{r}.npy"
                np.save(os.path.join(OUTDIR, fname_sent), sent)

    print("\nDone. All files saved to", OUTDIR)


if __name__ == "__main__":
    main()