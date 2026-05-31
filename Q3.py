import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.set_printoptions(precision=6, suppress=True)

# Clean, refactored CE586 HW4 script (Q1-Q4e)
# Preserves all formulas, constants, sign conventions, and numerical results.

# Given constants
EI = 3600.0
h = 3.0
Lb = 4.0
xi_target = 0.05

# Primary helper functions
def generalized_eigen(K, M):
    A = np.linalg.solve(M, K)
    eigvals, eigvecs = np.linalg.eig(A)
    eigvals = np.real(eigvals)
    eigvecs = np.real(eigvecs)
    idx = np.argsort(eigvals)
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    omega = np.sqrt(eigvals)
    periods = 2 * np.pi / omega
    Phi = eigvecs.copy()
    for n in range(Phi.shape[1]):
        Phi[:, n] = Phi[:, n] / Phi[2, n]
    return eigvals, omega, periods, Phi


def rayleigh_coefficients(omega1, omega2, xi=0.05):
    A = np.array([
        [1.0 / (2.0 * omega1), omega1 / 2.0],
        [1.0 / (2.0 * omega2), omega2 / 2.0]
    ])
    b = np.array([xi, xi])
    a0, a1 = np.linalg.solve(A, b)
    return a0, a1


def print_matrix(name, A, unit=""):
    print(f"\n{name} {unit}")
    print("-" * (len(name) + len(unit) + 1))
    print(A)


def print_summary_tables(title, tables: dict):
    """Print small grouped summary tables. tables is a dict of name->array_or_value."""
    print(f"\n{title}")
    for name, val in tables.items():
        print(f"\n{name}:")
        print(val)


def build_mass_matrix():
    return np.diag([20.0, 15.0, 15.0])


def build_story_stiffnesses(EI=3600.0, h=3.0):
    """Return the per-story stiffness coefficients used in the rigid-beam model."""
    k1 = 2.0 * (12.0 * (2.0 * EI) / h**3)
    k2 = 2.0 * (12.0 * (2.0 * EI) / h**3)
    k3 = 2.0 * (12.0 * (1.0 * EI) / h**3)
    return k1, k2, k3


def build_rigid_stiffness(EI=3600.0, h=3.0):
    k1, k2, k3 = build_story_stiffnesses(EI=EI, h=h)
    K_rigid = np.array([
        [k1 + k2, -k2,      0.0],
        [-k2,     k2 + k3, -k3],
        [0.0,     -k3,      k3]
    ])
    return K_rigid


def compute_rayleigh_damping(omega, xi_target=0.05, M=None, K=None):
    """If M and K provided, returns (a0,a1,C). Otherwise returns (a0,a1)."""
    a0, a1 = rayleigh_coefficients(omega[0], omega[1], xi=xi_target)
    if M is not None and K is not None:
        C = a0 * M + a1 * K
        return a0, a1, C
    return a0, a1


def build_finite_beam_condensed_stiffness(K_rigid=None):
    """If K_rigid is None, build default rigid stiffness and then condense."""
    if K_rigid is None:
        K_rigid = build_rigid_stiffness(EI=EI, h=h)
    Kdd = K_rigid.copy()
    Kds = np.array([
        [    0.0,     0.0,  4800.0,  4800.0,     0.0,     0.0],
        [-4800.0, -4800.0, -2400.0, -2400.0,  2400.0,  2400.0],
        [    0.0,     0.0, -2400.0, -2400.0, -2400.0, -2400.0]
    ])
    Ksd = Kds.T
    Kss = np.array([
        [22800.0,  1800.0,  4800.0,     0.0,     0.0,     0.0],
        [ 1800.0, 22800.0,     0.0,  4800.0,     0.0,     0.0],
        [ 4800.0,     0.0, 18000.0,  1800.0,  2400.0,     0.0],
        [    0.0,  4800.0,  1800.0, 18000.0,     0.0,  2400.0],
        [    0.0,     0.0,  2400.0,     0.0,  8400.0,  1800.0],
        [    0.0,     0.0,     0.0,  2400.0,  1800.0,  8400.0]
    ])
    K_condensed = Kdd - Kds @ np.linalg.solve(Kss, Ksd)
    return Kdd, Kds, Kss, K_condensed


def print_stiffness_matrix_formation(EI=3600.0, h=3.0):
    """Print the intermediate stiffness matrices used in Q1."""
    k1, k2, k3 = build_story_stiffnesses(EI=EI, h=h)
    print("\nQ1(a) Stiffness matrix formation")
    print("--------------------------------")
    print(f"Story stiffness coefficients:\n  k1 = {k1:.3f} kN/m\n  k2 = {k2:.3f} kN/m\n  k3 = {k3:.3f} kN/m")

    K_rigid = build_rigid_stiffness(EI=EI, h=h)
    print_matrix("Rigid-beam stiffness matrix K_rigid", K_rigid, "(kN/m)")

    Kdd, Kds, Kss, K_condensed = build_finite_beam_condensed_stiffness(K_rigid)
    print_matrix("Finite-beam partition K_dd", Kdd, "(kN/m)")
    print_matrix("Finite-beam partition K_ds", Kds, "(kN/m)")
    print_matrix("Finite-beam partition K_ss", Kss, "(kN/m)")
    print_matrix("Condensed stiffness matrix K_c", K_condensed, "(kN/m)")

    return {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "K_rigid": K_rigid,
        "Kdd": Kdd,
        "Kds": Kds,
        "Kss": Kss,
        "K_condensed": K_condensed,
    }


def compute_modal_properties(Phi, M):
    ones = np.ones(Phi.shape[0])
    L_n = Phi.T @ M @ ones
    M_n = np.zeros(Phi.shape[1])
    for n in range(Phi.shape[1]):
        phi_n = Phi[:, n]
        M_n[n] = phi_n.T @ M @ phi_n
    Gamma_n = L_n / M_n
    M_star = L_n**2 / M_n
    mass_total = np.sum(np.diag(M))
    mass_participation = 100 * M_star / mass_total
    h = np.array([3.0, 6.0, 9.0])
    h_star = (Phi.T @ M @ h) / L_n
    return {
        'L_n': L_n,
        'M_n': M_n,
        'Gamma_n': Gamma_n,
        'M_star': M_star,
        'mass_total': mass_total,
        'mass_participation': mass_participation,
        'h_star': h_star
    }


def read_ground_motion(file_path: Path):
    if not file_path.exists():
        raise FileNotFoundError(f"Ground motion file not found: {file_path}")
    data = np.loadtxt(file_path)
    time = data[:, 0]
    ag_cm_s2 = data[:, 1]
    ag = ag_cm_s2 / 100.0
    dt = time[1] - time[0]
    return time, ag, dt, ag_cm_s2


def generalized_eigen_analysis(K, M):
    """Wrapper matching requested name: generalized_eigen_analysis(K, M)"""
    return generalized_eigen(K, M)


def newmark_sdof_base_excitation(time, ag, omega_n, xi_n):
    beta = 1 / 4
    gamma = 1 / 2
    dt = time[1] - time[0]
    n_steps = len(time)
    m = 1.0
    k = omega_n**2 * m
    c = 2 * xi_n * omega_n * m
    p = -m * ag
    D = np.zeros(n_steps)
    Ddot = np.zeros(n_steps)
    Dddot = np.zeros(n_steps)
    D[0] = 0.0
    Ddot[0] = 0.0
    Dddot[0] = (p[0] - c * Ddot[0] - k * D[0]) / m
    denom = m + gamma * dt * c + beta * dt**2 * k
    for i in range(n_steps - 1):
        D_pred = D[i] + dt * Ddot[i] + dt**2 * (0.5 - beta) * Dddot[i]
        Ddot_pred = Ddot[i] + dt * (1 - gamma) * Dddot[i]
        Dddot[i + 1] = (p[i + 1] - c * Ddot_pred - k * D_pred) / denom
        D[i + 1] = D_pred + beta * dt**2 * Dddot[i + 1]
        Ddot[i + 1] = Ddot_pred + gamma * dt * Dddot[i + 1]
    return D, Ddot, Dddot


def reconstruct_floor_displacements(Phi, Gamma, D_histories):
    modal_coords = Gamma[:, None] * D_histories
    u = Phi @ modal_coords
    return u


def modal_pseudo_acceleration_history(omega_n, D_histories):
    omega_n = np.asarray(omega_n, dtype=float)
    if omega_n.ndim != 1:
        raise ValueError("omega_n must be a 1D array of modal circular frequencies")
    return omega_n[:, None] ** 2 * np.asarray(D_histories, dtype=float)


def modal_effective_force_pattern(M, phi_n, Gamma_n):
    return Gamma_n * (M @ phi_n)


def modal_effective_force_patterns(M, Phi, Gamma):
    return np.column_stack([
        modal_effective_force_pattern(M, Phi[:, mode], Gamma[mode])
        for mode in range(Phi.shape[1])
    ])


def modal_floor_force_history(s_n, A_n):
    return np.asarray(s_n, dtype=float)[:, None] * np.asarray(A_n, dtype=float)[None, :]


def modal_story_shear_pattern(s_n):
    s_n = np.asarray(s_n, dtype=float)
    return np.flip(np.cumsum(np.flip(s_n)))


def modal_story_moment_pattern(s_n, h_floor):
    s_n = np.asarray(s_n, dtype=float)
    h_floor = np.asarray(h_floor, dtype=float)
    story_moment = np.zeros_like(s_n, dtype=float)
    for i in range(len(s_n)):
        story_moment[i] = np.sum((h_floor[i:] - h_floor[i]) * s_n[i:])
    return story_moment


def modal_base_shear_pattern(s_n):
    return float(np.sum(np.asarray(s_n, dtype=float)))


def modal_base_moment_pattern(s_n, h_floor):
    s_n = np.asarray(s_n, dtype=float)
    h_floor = np.asarray(h_floor, dtype=float)
    return float(np.sum(h_floor * s_n))


def modal_effective_height(s_n, h_floor):
    s_n = np.asarray(s_n, dtype=float)
    h_floor = np.asarray(h_floor, dtype=float)
    return modal_base_moment_pattern(s_n, h_floor) / modal_base_shear_pattern(s_n)


def modal_base_shear_history(s_n, A_n):
    return modal_base_shear_pattern(s_n) * np.asarray(A_n, dtype=float)


def modal_base_moment_history(s_n, h_floor, A_n):
    return modal_base_moment_pattern(s_n, h_floor) * np.asarray(A_n, dtype=float)


def compute_drifts_shears_moment(u, k_story, h_floor):
    n_floors = u.shape[0]
    drifts = np.zeros_like(u)

    for i in range(n_floors):
        if i == 0:
            drifts[i, :] = u[i, :]
        else:
            drifts[i, :] = u[i, :] - u[i - 1, :]

    story_shears = k_story[:, None] * drifts

    # Base shear is the first-story shear.
    base_shear = story_shears[0, :]

    # Decompose story shears into per-floor lateral forces f_i(t):
    # For a 3-story system: V1 = f1+f2+f3, V2 = f2+f3, V3 = f3
    # therefore f3 = V3; f2 = V2 - V3; f1 = V1 - V2
    n_steps = story_shears.shape[1]
    floor_forces = np.zeros_like(story_shears)
    # compute per-time-step decomposition
    for j in range(n_steps):
        V1_j = story_shears[0, j]
        V2_j = story_shears[1, j]
        V3_j = story_shears[2, j]
        f3 = V3_j
        f2 = V2_j - V3_j
        f1 = V1_j - V2_j
        floor_forces[0, j] = f1
        floor_forces[1, j] = f2
        floor_forces[2, j] = f3

    # Base overturning moment: sum over floors of h_i * f_i(t)
    base_moment = np.sum(h_floor[:, None] * floor_forces, axis=0)

    return drifts, story_shears, base_shear, base_moment, floor_forces


def save_results(filename, arr, header=None):
    np.savetxt(filename, arr, delimiter=",", header=header or "", comments="")


def get_abs_max(time, response):
    idx = np.argmax(np.abs(response))
    return {
        "max_abs": np.abs(response[idx]),
        "time": time[idx],
        "signed": response[idx]
    }


# Main execution wrapped in a function
def main():
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    figures_dir = script_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    for stale_name in [
        "Q3c_deflections_drifts_shears.csv",
        "Q3cdef_RHA_results.csv",
        "Q4c_RHA_RSA_comparison.csv",
        "Q4c_RHA_RSA_comparison_table.tex",
    ]:
        (data_dir / stale_name).unlink(missing_ok=True)

    # Build M and K for Q1/Q2
    M = build_mass_matrix()
    stiffness_info = print_stiffness_matrix_formation(EI=EI, h=h)
    K_rigid = stiffness_info["K_rigid"]

    lambda_r, omega_r, T_r, Phi_r = generalized_eigen(K_rigid, M)
    a0_r, a1_r, C_rigid = compute_rayleigh_damping(omega_r, xi_target=xi_target, M=M, K=K_rigid)

    Kdd = stiffness_info["Kdd"]
    Kds = stiffness_info["Kds"]
    Kss = stiffness_info["Kss"]
    K_condensed = stiffness_info["K_condensed"]
    lambda_c, omega_c, T_c, Phi_c = generalized_eigen(K_condensed, M)
    a0_c, a1_c, C_condensed = compute_rayleigh_damping(omega_c, xi_target=xi_target, M=M, K=K_condensed)

    # Print Q1 summary
    print("\n" + "=" * 70)
    print("Q1(a) FLEXURALLY RIGID BEAMS")
    print("=" * 70)
    print_matrix("Mass matrix M", M, "(t)")
    print_matrix("Rigid-beam damping matrix C_rigid", C_rigid, "(kN s/m)")

    # Save Q1/Q2 outputs (same filenames preserved)
    np.savetxt(data_dir / "Q1_K_rigid.csv", K_rigid, delimiter=",")
    np.savetxt(data_dir / "Q1_C_rigid.csv", C_rigid, delimiter=",")
    np.savetxt(data_dir / "Q1_K_condensed_finite_beam.csv", K_condensed, delimiter=",")
    np.savetxt(data_dir / "Q1_C_condensed_finite_beam.csv", C_condensed, delimiter=",")
    np.savetxt(data_dir / "Q2_mode_shapes_roof_normalized.csv", Phi_r, delimiter=",")

    q2_summary = np.column_stack((np.arange(1,4), lambda_r, omega_r, T_r))
    np.savetxt(data_dir / "Q2_eigen_summary.csv", q2_summary, delimiter=",", header="mode,lambda_omega_squared,omega_rad_s,period_s", comments="")

    # Q3(a): modal properties from provided K
    K_q3 = np.array([
        [12800.0, -6400.0,     0.0],
        [-6400.0,  9600.0, -3200.0],
        [    0.0, -3200.0,  3200.0]
    ])
    lambda_vals, omega_vals, periods_vals, Phi_q3 = generalized_eigen(K_q3, M)
    modal = compute_modal_properties(Phi_q3, M)

    print("\nQ3(a) Modal properties")
    print("Mode | L_n       | M_n       | Gamma_n")
    for n in range(3):
        print(f"{n+1:>4d} | {modal['L_n'][n]:>9.3f} | {modal['M_n'][n]:>9.3f} | {modal['Gamma_n'][n]:>8.3f}")

    print("\nUseful for Q3(f)")
    print("Mode | M*_n      | Mass %    | h*_n")
    for n in range(3):
        print(f"{n+1:>4d} | {modal['M_star'][n]:>9.3f} | {modal['mass_participation'][n]:>8.2f} | {modal['h_star'][n]:>8.3f}")
    print(f"\nSum of effective modal masses = {np.sum(modal['M_star']):.3f} t")
    print(f"Total physical mass = {modal['mass_total']:.3f} t")

    # Q3(b): read ground motion from script directory
    gm_file = script_dir / "data" / "DIN95Y01.THF"
    time, ag, dt, ag_cm_s2 = read_ground_motion(gm_file)
    print(f"Number of points: {len(time)}")
    print(f"Time step dt: {dt:.5f} s")
    print(f"Duration: {time[-1]:.2f} s")
    print(f"PGA: {np.max(np.abs(ag_cm_s2)):.3f} cm/s^2")
    print(f"PGA: {np.max(np.abs(ag)) / 9.80665:.4f} g")

    # Modal frequencies used in original script
    omega_modal = np.array([8.57354217, 19.53728262, 32.21990528])
    periods = 2 * np.pi / omega_modal

    a0, a1 = rayleigh_coefficients(omega_modal[0], omega_modal[1], xi=xi_target)
    xi_modal = a0 / (2 * omega_modal) + a1 * omega_modal / 2
    print("\nRayleigh damping coefficients:")
    print(f"a0 = {a0:.6f}")
    print(f"a1 = {a1:.6f}")
    print("\nModal damping ratios:")
    for i in range(3):
        print(f"Mode {i+1}: xi = {xi_modal[i]:.6f}")

    # Solve modal SDOF systems
    D_histories = []
    results = []
    for mode in range(3):
        D, Ddot, Dddot = newmark_sdof_base_excitation(time, ag, omega_modal[mode], xi_modal[mode])
        D_histories.append(D)
        idx_max = np.argmax(np.abs(D))
        results.append({
            'mode': mode+1,
            'T_s': periods[mode],
            'omega_rad_s': omega_modal[mode],
            'xi': xi_modal[mode],
            'max_abs_D_m': np.abs(D[idx_max]),
            'time_at_max_s': time[idx_max],
            'signed_D_at_max_m': D[idx_max]
        })
    D_histories = np.array(D_histories)

    print("\nQ3(b) Results: Modal SDOF displacement histories")
    print("Mode | T (s)  | xi     | max|D| (m) | time (s) | signed D (m)")
    for r in results:
        print(f"{r['mode']:>4d} | {r['T_s']:>6.3f} | {r['xi']:>6.4f} | {r['max_abs_D_m']:>10.5f} | {r['time_at_max_s']:>8.2f} | {r['signed_D_at_max_m']:>12.5f}")

    # Plot modal SDOF displacement histories for Q3(b)
    plt.figure(figsize=(10, 6))
    for mode in range(3):
        plt.plot(
            time,
            D_histories[mode],
            linewidth=1.2,
            label=f"Mode {mode+1}: T = {periods[mode]:.3f} s"
        )
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Time, t (s)")
    plt.ylabel("Equivalent SDOF displacement, $D_n(t)$ (m)")
    plt.title("Q3(b) Modal SDOF Displacement Histories - DIN95Y01")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Q3b_modal_SDOF_displacement_histories.png", dpi=300)
    plt.show()

    # Save modal histories
    out = np.column_stack((time, D_histories[0], D_histories[1], D_histories[2]))
    save_results(data_dir / "Q3b_modal_D_histories.csv", out, header="time_s,D1_m,D2_m,D3_m")

    Gamma = modal['Gamma_n']
    modal_coefficients = Phi_q3 * Gamma[None, :]
    print("\nQ3(b) Modal displacement equations")
    for mode in range(3):
        coeff = modal_coefficients[:, mode]
        print(f"Mode {mode + 1}: phi_{mode + 1} * Gamma_{mode + 1} = [{coeff[0]:.6f}, {coeff[1]:.6f}, {coeff[2]:.6f}]^T")
        print(f"u_{mode + 1}(t) = {coeff[0]:.6f}D_1(t) + {coeff[1]:.6f}D_2(t) + {coeff[2]:.6f}D_3(t)")

    # Reconstruct floor displacements and compute drifts/shears/moments
    u = reconstruct_floor_displacements(Phi_q3, Gamma, D_histories)
    k_story = np.array([6400.0, 6400.0, 3200.0])
    h_floor = np.array([3.0, 6.0, 9.0])
    drifts, story_shears, base_shear, base_moment, floor_forces = compute_drifts_shears_moment(u, k_story, h_floor)

    # Save combined results
    out2 = np.column_stack((time, u[0,:], u[1,:], u[2,:], drifts[0,:], drifts[1,:], drifts[2,:], story_shears[0,:], story_shears[1,:], story_shears[2,:], base_shear, base_moment))
    header = ("time_s,u1_m,u2_m,u3_m,drift1_m,drift2_m,drift3_m,V1_kN,V2_kN,V3_kN,Vbase_kN,Mb_kNm")
    save_results(data_dir / "Q3c_deflections_drifts_shears.csv", out2, header=header)
    print(f"\nSaved: {(data_dir / 'Q3b_modal_D_histories.csv').name}")
    print(f"Saved: {(data_dir / 'Q3c_deflections_drifts_shears.csv').name}")

    # Print Q3(c)-(e)
    u1, u2, u3 = u[0,:], u[1,:], u[2,:]
    drift_1, drift_2, drift_3 = drifts[0,:], drifts[1,:], drifts[2,:]
    V1, V2, V3 = story_shears[0,:], story_shears[1,:], story_shears[2,:]
    Mb = base_moment
    V_base = V1

    print('\nQ3(c) Maximum floor displacements')
    for i, ui in enumerate([u1, u2, u3], start=1):
        r = get_abs_max(time, ui)
        print(f"u{i}: max|u{i}| = {r['max_abs']:.5f} m at t = {r['time']:.2f} s (signed = {r['signed']:.5f} m)")

    print('\nQ3(c) Maximum story drifts')
    for i in range(3):
        r = get_abs_max(time, drifts[i,:])
        print(f"Story {i+1}: max|Delta_{i+1}| = {r['max_abs']:.5f} m at t = {r['time']:.2f} s (signed = {r['signed']:.5f} m)")

    print('\nQ3(d) Maximum story shears')
    for i in range(3):
        r = get_abs_max(time, story_shears[i,:])
        print(f"V{i+1}: max|V{i+1}| = {r['max_abs']:.2f} kN at t = {r['time']:.2f} s (signed = {r['signed']:.2f} kN)")

    r_Vb = get_abs_max(time, V_base)
    r_Mb = get_abs_max(time, Mb)
    print('\nQ3(e) Base responses')
    print(f"Base shear: max|Vb| = {r_Vb['max_abs']:.2f} kN at t = {r_Vb['time']:.2f} s (signed = {r_Vb['signed']:.2f} kN)")
    print(f"Base overturning moment: max|Mb| = {r_Mb['max_abs']:.2f} kNm at t = {r_Mb['time']:.2f} s (signed = {r_Mb['signed']:.2f} kNm)")

    # Q3(f) effective modal base shear and moment vs direct RHA
    L_n = modal['L_n']
    M_n = modal['M_n']
    Gamma_n = modal['Gamma_n']
    M_star = modal['M_star']
    h_star = modal['h_star']
    Vbase_direct = V1
    Mb_direct = Mb
    # A_n = omega^2 * D_n
    A_modal = omega_modal[:, None]**2 * D_histories
    Vb_eff_total = np.sum(M_star[:, None] * A_modal, axis=0)
    Mb_eff_total = np.sum((M_star * h_star)[:, None] * A_modal, axis=0)

    r_Vb_eff = get_abs_max(time, Vb_eff_total)
    r_Mb_eff = get_abs_max(time, Mb_eff_total)

    print('\nQ3(f) Effective modal parameters')
    print('Mode | L_n      | M_n      | Gamma_n  | M*_n     | Mass %   | h*_n')
    for i in range(3):
        print(f"{i+1:>4d} | {L_n[i]:>8.3f} | {M_n[i]:>8.3f} | {Gamma_n[i]:>8.3f} | {M_star[i]:>8.3f} | {100.0*M_star[i]/modal['mass_total']:>7.2f} | {h_star[i]:>8.3f}")
    print(f"\nSum of effective modal masses = {np.sum(M_star):.3f} t")

    s_modal = modal_effective_force_patterns(M, Phi_q3, Gamma)
    modal_floor_forces = np.stack([
        modal_floor_force_history(s_modal[:, mode], A_modal[mode, :])
        for mode in range(3)
    ], axis=0)
    modal_story_shears = np.column_stack([
        modal_story_shear_pattern(s_modal[:, mode])
        for mode in range(3)
    ])
    modal_story_moments = np.column_stack([
        modal_story_moment_pattern(s_modal[:, mode], h_floor)
        for mode in range(3)
    ])
    modal_base_shear_history_from_forces = np.sum(modal_floor_forces, axis=1)
    modal_base_shear_history_from_mass = np.vstack([
        modal_base_shear_history(s_modal[:, mode], A_modal[mode, :])
        for mode in range(3)
    ])
    modal_base_moment_history_from_forces = np.vstack([
        modal_base_moment_history(s_modal[:, mode], h_floor, A_modal[mode, :])
        for mode in range(3)
    ])
    modal_base_moment_history_from_height = np.vstack([
        modal_effective_height(s_modal[:, mode], h_floor) * modal_base_shear_history_from_mass[mode, :]
        for mode in range(3)
    ])

    floor_heights = h_floor
    modal_base_moment_static_coefficients = np.array([
        np.sum(floor_heights * s_modal[:, mode])
        for mode in range(3)
    ])
    effective_modal_heights = np.array([
        modal_base_moment_static_coefficients[mode] / M_star[mode]
        for mode in range(3)
    ])
    base_moment_from_floor_force_sum = modal_base_moment_static_coefficients[:, None] * A_modal
    base_moment_from_effective_height = effective_modal_heights[:, None] * modal_base_shear_history_from_mass
    base_moment_verification = base_moment_from_floor_force_sum - base_moment_from_effective_height

    print('\nQ3(f) Modal overturning moment coefficients')
    print('Mode | M_bn^st   | M_bn(t) = M_bn^st A_n(t) | h*_n')
    for mode in range(3):
        print(f"{mode+1:>4d} | {modal_base_moment_static_coefficients[mode]:>9.6f} | {modal_base_moment_static_coefficients[mode]:>9.6f} A_{mode+1}(t) | {effective_modal_heights[mode]:>8.6f}")
    print('\nLaTeX-ready equations for Chapter 4.4:')
    for mode in range(3):
        print(f"M_b{mode+1}^{{st}} = {modal_base_moment_static_coefficients[mode]:.9f}")
        print(f"M_b{mode+1}(t) = {modal_base_moment_static_coefficients[mode]:.9f} A_{mode+1}(t)")
        print(f"h_{mode+1}^* = {effective_modal_heights[mode]:.9f}")
        print(f"M_b{mode+1}(t) = {effective_modal_heights[mode]:.9f} M_{mode+1}^* A_{mode+1}(t)")
        print(f"M_b{mode+1}^{{st}} - h_{mode+1}^* M_{mode+1}^* = {(modal_base_moment_static_coefficients[mode] - effective_modal_heights[mode] * M_star[mode]):.9e}")

    print('\nQ3(f) Base responses from effective modal parameters')
    print(f"Vb_eff: max|Vb| = {r_Vb_eff['max_abs']:.2f} kN at t = {r_Vb_eff['time']:.2f} s (signed = {r_Vb_eff['signed']:.2f} kN)")
    print(f"Mb_eff: max|Mb| = {r_Mb_eff['max_abs']:.2f} kNm at t = {r_Mb_eff['time']:.2f} s (signed = {r_Mb_eff['signed']:.2f} kNm)")
    print('\nQ3(f) Modal check values')
    print('Mode | sum(s_in)  | M*_n      | h*_n')
    for mode in range(3):
        base_shear_pattern = modal_base_shear_pattern(s_modal[:, mode])
        effective_height = modal_effective_height(s_modal[:, mode], h_floor)
        print(f"{mode+1:>4d} | {base_shear_pattern:>9.3f} | {M_star[mode]:>8.3f} | {effective_height:>6.3f}")
    print('\nMode-by-mode verification of response identities:')
    for mode in range(3):
        shear_error = np.max(np.abs(modal_base_shear_history_from_forces[mode, :] - modal_base_shear_history_from_mass[mode, :]))
        moment_error = np.max(np.abs(modal_base_moment_history_from_forces[mode, :] - modal_base_moment_history_from_height[mode, :]))
        print(f"Mode {mode+1}: max|Vb(force)-Vb(mass)| = {shear_error:.6e}, max|Mb(force)-Mb(height)| = {moment_error:.6e}")

    # Plot Q3(c) floor displacements
    plt.figure(figsize=(10, 6))
    plt.plot(time, u1, label="$u_1(t)$")
    plt.plot(time, u2, label="$u_2(t)$")
    plt.plot(time, u3, label="$u_3(t)$")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Time, t (s)")
    plt.ylabel("Floor displacement, $u_i(t)$ (m)")
    plt.title("Q3(c) Floor Displacement Response Histories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Q3c_floor_displacement_histories.png", dpi=300)
    plt.show()

    # Plot Q3(c) story drifts
    plt.figure(figsize=(10, 6))
    plt.plot(time, drift_1, label="$\\Delta_1=u_1$")
    plt.plot(time, drift_2, label="$\\Delta_2=u_2-u_1$")
    plt.plot(time, drift_3, label="$\\Delta_3=u_3-u_2$")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Time, t (s)")
    plt.ylabel("Story drift, $\\Delta_i(t)$ (m)")
    plt.title("Q3(c) Story Drift Histories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Q3c_story_drift_histories.png", dpi=300)
    plt.show()

    # Plot Q3(d) story shears
    plt.figure(figsize=(10, 6))
    plt.plot(time, V1, label="$V_1(t)$")
    plt.plot(time, V2, label="$V_2(t)$")
    plt.plot(time, V3, label="$V_3(t)$")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Time, t (s)")
    plt.ylabel("Story shear, $V_i(t)$ (kN)")
    plt.title("Q3(d) Story Shear Response Histories")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Q3d_story_shear_histories.png", dpi=300)
    plt.show()

    # Plot Q3(e) base overturning moment
    plt.figure(figsize=(10, 6))
    plt.plot(time, Mb, label="$M_b(t)$")
    plt.axhline(0, linewidth=0.8)
    plt.xlabel("Time, t (s)")
    plt.ylabel("Base overturning moment, $M_b(t)$ (kNm)")
    plt.title("Q3(e) Base Overturning Moment History")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Q3e_base_overturning_moment_history.png", dpi=300)
    plt.show()

    # Verification prints
    print('\nVerification:')
    print(f"Sum of effective modal masses = {np.sum(M_star):.6f} t")
    print(f"Total mass = {modal['mass_total']:.6f} t")
    Vbase_eff = Vb_eff_total
    Mb_eff = Mb_eff_total
    print(f"Max abs difference in base shear = {np.max(np.abs(Vbase_direct - Vbase_eff)):.6e}")
    print(f"Max abs difference in overturning moment = {np.max(np.abs(Mb_direct - Mb_eff)):.6e}")

    # Save final RHA results (preserve filenames)
    out_final = np.column_stack((time, u[0,:], u[1,:], u[2,:], drifts[0,:], drifts[1,:], drifts[2,:], story_shears[0,:], story_shears[1,:], story_shears[2,:], V_base, Mb, Vbase_eff, Mb_eff))
    save_results(data_dir / "Q3cdef_RHA_results.csv", out_final, header="time_s,u1_m,u2_m,u3_m,drift1_m,drift2_m,drift3_m,V1_kN,V2_kN,V3_kN,Vbase_kN,Mb_kNm,Vbase_eff_kN,Mb_eff_kNm")
    print('\nSaved plots and CSV:')
    print((figures_dir / 'Q3b_modal_SDOF_displacement_histories.png').name)
    print((data_dir / 'Q3b_modal_D_histories.csv').name)
    print((figures_dir / 'Q3c_floor_displacement_histories.png').name)
    print((figures_dir / 'Q3c_story_drift_histories.png').name)
    print((figures_dir / 'Q3d_story_shear_histories.png').name)
    print((figures_dir / 'Q3e_base_overturning_moment_history.png').name)
    print((data_dir / 'Q3c_deflections_drifts_shears.csv').name)
    print((data_dir / 'Q3cdef_RHA_results.csv').name)

    # Q4(a) through Q4(e)
    q4a_response_spectrum(script_dir)
    q4b_rsa(script_dir)
    q4c_compare(script_dir)
    q4d_eslfp(script_dir)
    q4e_compare(script_dir)


def q4a_response_spectrum(script_dir: Path):
    data_dir = script_dir / "data"
    figures_dir = script_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data = np.loadtxt(script_dir / "data" / "DIN95Y01.THF")
    time = data[:, 0]
    ag_cm_s2 = data[:, 1]
    ag = ag_cm_s2 / 100.0
    dt = time[1] - time[0]
    print(f"Number of points: {len(time)}")
    print(f"Time step: {dt:.5f} s")
    print(f"Duration: {time[-1]:.2f} s")
    print(f"PGA: {np.max(np.abs(ag_cm_s2)):.3f} cm/s^2")
    print(f"PGA: {np.max(np.abs(ag)) / 9.80665:.4f} g")

    def newmark_sdof_base_excitation(time, ag, T, xi=0.05):
        beta = 1.0 / 4.0
        gamma = 1.0 / 2.0
        dt = time[1] - time[0]
        n = len(time)
        omega = 2.0 * np.pi / T
        m = 1.0
        k = m * omega**2
        c = 2.0 * xi * m * omega
        p = -m * ag
        u = np.zeros(n)
        udot = np.zeros(n)
        uddot = np.zeros(n)
        uddot[0] = (p[0] - c * udot[0] - k * u[0]) / m
        denom = m + gamma * dt * c + beta * dt**2 * k
        for i in range(n - 1):
            u_pred = u[i] + dt * udot[i] + dt**2 * (0.5 - beta) * uddot[i]
            udot_pred = udot[i] + dt * (1.0 - gamma) * uddot[i]
            uddot[i + 1] = (p[i + 1] - c * udot_pred - k * u_pred) / denom
            u[i + 1] = u_pred + beta * dt**2 * uddot[i + 1]
            udot[i + 1] = udot_pred + gamma * dt * uddot[i + 1]
        return u, udot, uddot

    T_values = np.arange(0.05, 3.00 + 0.01, 0.01)
    Sd = np.zeros_like(T_values)
    Sa_pseudo = np.zeros_like(T_values)
    Sa_pseudo_g = np.zeros_like(T_values)
    Sa_actual = np.zeros_like(T_values)
    Sa_actual_g = np.zeros_like(T_values)
    for i, T in enumerate(T_values):
        omega = 2.0 * np.pi / T
        u, udot, uddot = newmark_sdof_base_excitation(time=time, ag=ag, T=T, xi=0.05)
        Sd[i] = np.max(np.abs(u))
        Sa_pseudo[i] = omega**2 * Sd[i]
        Sa_pseudo_g[i] = Sa_pseudo[i] / 9.80665
        abs_acc = uddot + ag
        Sa_actual[i] = np.max(np.abs(abs_acc))
        Sa_actual_g[i] = Sa_actual[i] / 9.80665

    modal_periods = np.array([0.732858, 0.321600, 0.195009])
    Sa_modal_g = np.interp(modal_periods, T_values, Sa_pseudo_g)
    Sd_modal = np.interp(modal_periods, T_values, Sd)

    print("\n5% elastic spectrum ordinates at structural modal periods")
    print("---------------------------------------------------------")
    print("Mode | T_n (s) | Sd (m)   | Sa,pseudo (g)")
    for i in range(3):
        print(f"{i+1:>4d} | {modal_periods[i]:>7.3f} | {Sd_modal[i]:>8.5f} | {Sa_modal_g[i]:>12.5f}")

    plt.figure(figsize=(10, 6))
    plt.plot(T_values, Sa_pseudo_g, linewidth=1.5, label="5% pseudo-acceleration spectrum")
    plt.scatter(modal_periods, Sa_modal_g, zorder=5, label="Modal periods")
    for i, Tn in enumerate(modal_periods):
        plt.annotate(f"Mode {i+1}\nT={Tn:.3f}s", xy=(Tn, Sa_modal_g[i]), xytext=(Tn + 0.05, Sa_modal_g[i] + 0.05), arrowprops=dict(arrowstyle="->", linewidth=0.8), fontsize=9)
    plt.xlabel("Period, T (s)")
    plt.ylabel("Pseudo-acceleration, $S_a$ (g)")
    plt.title("Q4(a) 5% Elastic Pseudo-Acceleration Response Spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Q4a_5percent_elastic_pseudo_acceleration_spectrum.png", dpi=300)
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(T_values, Sd, linewidth=1.5, label="5% displacement spectrum")
    plt.scatter(modal_periods, Sd_modal, zorder=5, label="Modal periods")
    plt.xlabel("Period, T (s)")
    plt.ylabel("Spectral displacement, $S_d$ (m)")
    plt.title("Q4(a) 5% Elastic Displacement Response Spectrum")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "Q4a_5percent_elastic_displacement_spectrum.png", dpi=300)
    plt.show()

    spectrum_output = np.column_stack((T_values, Sd, Sa_pseudo, Sa_pseudo_g, Sa_actual, Sa_actual_g))
    np.savetxt(data_dir / "Q4a_5percent_elastic_response_spectrum.csv", spectrum_output, delimiter=",", header="T_s,Sd_m,Sa_pseudo_m_per_s2,Sa_pseudo_g,Sa_actual_m_per_s2,Sa_actual_g", comments="")
    modal_output = np.column_stack((np.arange(1, 4), modal_periods, Sd_modal, Sa_modal_g))
    np.savetxt(data_dir / "Q4a_modal_period_spectral_values.csv", modal_output, delimiter=",", header="mode,T_s,Sd_m,Sa_pseudo_g", comments="")
    print("\nSaved:")
    print("Q4a_5percent_elastic_pseudo_acceleration_spectrum.png")
    print("Q4a_5percent_elastic_displacement_spectrum.png")
    print("Q4a_5percent_elastic_response_spectrum.csv")
    print("Q4a_modal_period_spectral_values.csv")


def q4b_rsa(script_dir: Path):
    data_dir = script_dir / "data"
    figures_dir = script_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    Phi = np.array([[0.37024474, -0.97779098, 3.10754624], [0.65544238, -0.78924412, -3.86619826], [1.0, 1.0, 1.0]])
    Gamma = np.array([1.33287597, -0.37718726, 0.04431129])
    omega = np.array([8.57354217, 19.53728262, 32.21990528])
    T_modal = 2.0 * np.pi / omega
    k_story = np.array([6400.0, 6400.0, 3200.0])
    xi_cqc = 0.05
    spectrum_file = data_dir / "Q4a_5percent_elastic_response_spectrum.csv"
    if not spectrum_file.exists():
        raise FileNotFoundError(f"Could not find {spectrum_file}\nRun the Q4(a) spectrum script first, or place Q4a_5percent_elastic_response_spectrum.csv in the same folder.")
    spectrum = np.genfromtxt(spectrum_file, delimiter=",", names=True)
    T_spectrum = spectrum["T_s"]
    Sa_pseudo_g = spectrum["Sa_pseudo_g"]
    g = 9.80665
    Sa_pseudo = Sa_pseudo_g * g
    if np.min(T_modal) < np.min(T_spectrum) or np.max(T_modal) > np.max(T_spectrum):
        raise ValueError(f"At least one modal period is outside the response spectrum period range.\nModal periods: {T_modal}\nSpectrum range: {T_spectrum[0]} to {T_spectrum[-1]} s\nRecompute Q4(a) with a wider period range, e.g. 0.05 to 3.00 s.")
    Sa_modal = np.interp(T_modal, T_spectrum, Sa_pseudo)
    Sa_modal_g = Sa_modal / g
    Sd_modal = Sa_modal / omega**2
    print("\nSpectral ordinates at modal periods")
    print("-----------------------------------")
    print("Mode | T_n (s) | Sa (g)   | Sa (m/s^2) | Sd (m)")
    for n in range(3):
        print(f"{n+1:>4d} | {T_modal[n]:>7.4f} | {Sa_modal_g[n]:>8.4f} | {Sa_modal[n]:>10.4f} | {Sd_modal[n]:>8.5f}")

    modal_u = Phi * (Gamma * Sd_modal)[None, :]
    modal_drift = np.zeros_like(modal_u)
    modal_drift[0, :] = modal_u[0, :]
    modal_drift[1, :] = modal_u[1, :] - modal_u[0, :]
    modal_drift[2, :] = modal_u[2, :] - modal_u[1, :]
    modal_V = k_story[:, None] * modal_drift
    # Decompose modal story shears into per-floor modal forces f_i,n:
    # V1 = f1+f2+f3, V2 = f2+f3, V3 = f3  => f3=V3, f2=V2-V3, f1=V1-V2
    h_floor = np.array([3.0, 6.0, 9.0])
    modal_floor_forces = np.zeros_like(modal_V)
    modal_floor_forces[2, :] = modal_V[2, :]
    modal_floor_forces[1, :] = modal_V[1, :] - modal_V[2, :]
    modal_floor_forces[0, :] = modal_V[0, :] - modal_V[1, :]
    modal_Mb = np.sum(h_floor[:, None] * modal_floor_forces, axis=0)
    print("\nModal floor displacement contributions, u_i,n (m)")
    print("-------------------------------------------------")
    print("        Mode 1      Mode 2      Mode 3")
    for i in range(3):
        print(f"u{i+1}: {modal_u[i,0]:>10.5f} {modal_u[i,1]:>10.5f} {modal_u[i,2]:>10.5f}")
    print("\nModal story shear contributions, V_i,n (kN)")
    print("-------------------------------------------")
    print("        Mode 1      Mode 2      Mode 3")
    for i in range(3):
        print(f"V{i+1}: {modal_V[i,0]:>10.2f} {modal_V[i,1]:>10.2f} {modal_V[i,2]:>10.2f}")
    print("\nModal base overturning moment contributions, Mb_n (kNm)")
    print("-------------------------------------------------------")
    for n in range(3):
        print(f"Mode {n+1}: Mb_{n+1} = {modal_Mb[n]:.2f} kNm")

    # --- New: carry forward modal maxima from Q3(b) and compute modal Vb/Mb from D_n maxima ---
    # Attempt to read modal SDOF histories produced by Q3(b) to get D_n maxima
    D_hist_file = Path(script_dir) / "data" / "Q3b_modal_D_histories.csv"
    Vb_from_D = np.full(3, np.nan)
    Mb_from_D = np.full(3, np.nan)
    try:
        if D_hist_file.exists():
            Ddata = np.genfromtxt(D_hist_file, delimiter=",", skip_header=1)
            # columns: time, D1, D2, D3
            Dcols = Ddata[:, 1:4]
            Dmax = np.max(np.abs(Dcols), axis=0)
            # compute modal effective masses and heights from Phi and Gamma
            M_local = np.diag([20.0, 15.0, 15.0])
            ones = np.ones(3)
            L_n = Phi.T @ M_local @ ones
            M_n = np.array([Phi[:, i].T @ M_local @ Phi[:, i] for i in range(3)])
            M_star_local = L_n**2 / M_n
            h_star_local = (Phi.T @ M_local @ h_floor) / L_n
            A_from_D = (omega**2) * Dmax
            Vb_from_D = M_star_local * A_from_D
            Mb_from_D = h_star_local * Vb_from_D
            print("\nCarried-forward modal maxima from Q3(b): D_n (m)", Dmax)
            for n in range(3):
                print(f"Mode {n+1}: Vb_from_D = {Vb_from_D[n]:.6f} kN, Mb_from_D = {Mb_from_D[n]:.6f} kNm")
        else:
            print(f"\nWarning: {D_hist_file} not found — skipping D-based modal V/M computation.")
    except Exception as e:
        print(f"\nWarning: failed to compute D-based modal V/M: {e}")

    def cqc_correlation_matrix(omega, xi=0.05):
        n_modes = len(omega)
        rho = np.eye(n_modes)
        for i in range(n_modes):
            for j in range(n_modes):
                if i != j:
                    beta = omega[j] / omega[i]
                    numerator = 8.0 * xi**2 * (1.0 + beta) * beta**1.5
                    denominator = (1.0 - beta**2)**2 + 4.0 * xi**2 * beta * (1.0 + beta)**2
                    rho[i, j] = numerator / denominator
        return rho

    def combine_modal_response(R_modal, rho):
        R_modal = np.asarray(R_modal, dtype=float)
        abssum = np.sum(np.abs(R_modal))
        srss = np.sqrt(np.sum(R_modal**2))
        cqc = np.sqrt(max(R_modal @ rho @ R_modal, 0.0))
        return abssum, srss, cqc

    rho_cqc = cqc_correlation_matrix(omega, xi=xi_cqc)
    print("\nCQC correlation matrix")
    print("----------------------")
    print(rho_cqc)
    u_results = np.array([[i + 1, *combine_modal_response(modal_u[i, :], rho_cqc)] for i in range(3)])
    V_results = np.array([[i + 1, *combine_modal_response(modal_V[i, :], rho_cqc)] for i in range(3)])
    Mb_abssum, Mb_srss, Mb_cqc = combine_modal_response(modal_Mb, rho_cqc)
    print("\nQ4(b) RSA floor displacement maxima")
    print("-----------------------------------")
    print("Floor | ABSSUM (m) | SRSS (m) | CQC (m)")
    for row in u_results:
        print(f"{int(row[0]):>5d} | {row[1]:>10.5f} | {row[2]:>8.5f} | {row[3]:>7.5f}")
    print("\nQ4(b) RSA story shear maxima")
    print("----------------------------")
    print("Story | ABSSUM (kN) | SRSS (kN) | CQC (kN)")
    for row in V_results:
        print(f"{int(row[0]):>5d} | {row[1]:>11.2f} | {row[2]:>9.2f} | {row[3]:>8.2f}")
    print("\nQ4(b) RSA base overturning moment maximum")
    print("-----------------------------------------")
    print("Rule   | Mb (kNm)")
    print(f"ABSSUM | {Mb_abssum:.2f}")
    print(f"SRSS   | {Mb_srss:.2f}")
    print(f"CQC    | {Mb_cqc:.2f}")
    np.savetxt(data_dir / "Q4b_RSA_floor_displacements.csv", u_results, delimiter=",", header="floor,u_ABSSUM_m,u_SRSS_m,u_CQC_m", comments="")
    np.savetxt(data_dir / "Q4b_RSA_story_shears.csv", V_results, delimiter=",", header="story,V_ABSSUM_kN,V_SRSS_kN,V_CQC_kN", comments="")
    with open(data_dir / "Q4b_RSA_base_overturning_moment.csv", "w") as f:
        f.write("rule,Mb_kNm\n")
        for rule, value in np.array([["ABSSUM", Mb_abssum], ["SRSS", Mb_srss], ["CQC", Mb_cqc]], dtype=object):
            f.write(f"{rule},{float(value):.6f}\n")
    modal_values = np.column_stack((np.arange(1, 4), T_modal, Sa_modal_g, Sa_modal, Sd_modal, modal_Mb, Vb_from_D, Mb_from_D))
    np.savetxt(data_dir / "Q4b_RSA_modal_values.csv", modal_values, delimiter=",", header="mode,T_s,Sa_g,Sa_m_per_s2,Sd_m,Mb_modal_kNm,Vb_from_D_kN,Mb_from_D_kNm", comments="")
    print("\nSaved:")
    print("Q4b_RSA_floor_displacements.csv")
    print("Q4b_RSA_story_shears.csv")
    print("Q4b_RSA_base_overturning_moment.csv")
    print("Q4b_RSA_modal_values.csv")


def q4c_compare(script_dir: Path):
    import pandas as pd
    data_dir = script_dir / "data"
    figures_dir = script_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rha_file = data_dir / "Q3cdef_RHA_results.csv"
    rsa_u_file = data_dir / "Q4b_RSA_floor_displacements.csv"
    rsa_v_file = data_dir / "Q4b_RSA_story_shears.csv"
    rsa_mb_file = data_dir / "Q4b_RSA_base_overturning_moment.csv"
    for f in [rha_file, rsa_u_file, rsa_v_file, rsa_mb_file]:
        if not f.exists():
            raise FileNotFoundError(f"Missing required file: {f}\nMake sure Q3(c)-(f) and Q4(b) scripts have been run first.")
    rha = pd.read_csv(rha_file)
    rha_values = {"u1": np.max(np.abs(rha["u1_m"])), "u2": np.max(np.abs(rha["u2_m"])), "u3": np.max(np.abs(rha["u3_m"])), "V1": np.max(np.abs(rha["V1_kN"])), "V2": np.max(np.abs(rha["V2_kN"])), "V3": np.max(np.abs(rha["V3_kN"])), "Mb": np.max(np.abs(rha["Mb_kNm"]))}
    units = {"u1": "m", "u2": "m", "u3": "m", "V1": "kN", "V2": "kN", "V3": "kN", "Mb": "kNm"}
    rsa_u = pd.read_csv(rsa_u_file); rsa_v = pd.read_csv(rsa_v_file); rsa_mb = pd.read_csv(rsa_mb_file)
    rsa_values = {}
    for _, row in rsa_u.iterrows(): rsa_values[f"u{int(row['floor'])}"] = {"ABSSUM": row["u_ABSSUM_m"], "SRSS": row["u_SRSS_m"], "CQC": row["u_CQC_m"]}
    for _, row in rsa_v.iterrows(): rsa_values[f"V{int(row['story'])}"] = {"ABSSUM": row["V_ABSSUM_kN"], "SRSS": row["V_SRSS_kN"], "CQC": row["V_CQC_kN"]}
    rsa_values["Mb"] = {"ABSSUM": float(rsa_mb.loc[rsa_mb["rule"] == "ABSSUM", "Mb_kNm"].iloc[0]), "SRSS": float(rsa_mb.loc[rsa_mb["rule"] == "SRSS", "Mb_kNm"].iloc[0]), "CQC": float(rsa_mb.loc[rsa_mb["rule"] == "CQC", "Mb_kNm"].iloc[0])}
    def percent_difference(approx, reference): return 100.0 * (approx - reference) / reference
    response_labels = {"u1": "Floor displacement u1", "u2": "Floor displacement u2", "u3": "Floor displacement u3", "V1": "Story shear V1", "V2": "Story shear V2", "V3": "Story shear V3", "Mb": "Base overturning moment Mb"}
    rows = []
    for key in ["u1", "u2", "u3", "V1", "V2", "V3", "Mb"]:
        rha_val = rha_values[key]
        rows.append({"Response": response_labels[key], "Unit": units[key], "RHA": rha_val, "RSA_ABSSUM": rsa_values[key]["ABSSUM"], "ABSSUM_Diff_%": percent_difference(rsa_values[key]["ABSSUM"], rha_val), "RSA_SRSS": rsa_values[key]["SRSS"], "SRSS_Diff_%": percent_difference(rsa_values[key]["SRSS"], rha_val), "RSA_CQC": rsa_values[key]["CQC"], "CQC_Diff_%": percent_difference(rsa_values[key]["CQC"], rha_val)})
    comparison = pd.DataFrame(rows)
    display_table = comparison.copy()
    for col in ["RHA", "RSA_ABSSUM", "RSA_SRSS", "RSA_CQC"]: display_table[col] = display_table[col].round(5)
    for col in ["ABSSUM_Diff_%", "SRSS_Diff_%", "CQC_Diff_%"]: display_table[col] = display_table[col].round(2)
    print("\nQ4(c) RHA vs RSA Comparison Table")
    print("=================================")
    print(display_table.to_string(index=False))
    output_file = data_dir / "Q4c_RHA_RSA_comparison.csv"; comparison.to_csv(output_file, index=False)
    print(f"\nSaved: {output_file.name}")
    latex_file = script_dir / "Q4c_RHA_RSA_comparison_table.tex"
    try:
        latex_table = display_table.to_latex(index=False, float_format="%.3f", caption="Comparison of RHA and RSA response quantities.", label="tab:rha_rsa_comparison")
    except Exception:
        latex_table = None
        print("Skipped LaTeX export for Q4(c) because the optional jinja2 dependency is not installed.")
    if latex_table is not None:
        with open(latex_file, "w", encoding="utf-8") as f:
            f.write(latex_table)
        print(f"Saved: {latex_file.name}")
    print("\nShort interpretation")
    print("--------------------")
    for rule in ["ABSSUM", "SRSS", "CQC"]:
        diff_col = f"{rule}_Diff_%"; mean_abs_diff = np.mean(np.abs(comparison[diff_col])); max_abs_diff = np.max(np.abs(comparison[diff_col])); print(f"{rule}: mean absolute difference = {mean_abs_diff:.2f}%, maximum absolute difference = {max_abs_diff:.2f}%")
    print("\nGeneral conclusion: ABSSUM is expected to be conservative because it adds absolute modal maxima. SRSS and CQC should be closer to RHA because they account for the statistical combination of modal maxima. CQC becomes especially important for closely spaced modes.")


def q4d_eslfp(script_dir: Path):
    data_dir = script_dir / "data"
    figures_dir = script_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    g = 9.80665
    T1 = 0.732858
    m = np.array([20.0, 15.0, 15.0])
    h = np.array([3.0, 6.0, 9.0])
    def design_spectrum_A(T):
        if T <= 0.15: return 0.4 + 0.6 * (T / 0.15)
        elif T <= 0.60: return 1.0
        return (0.6 / T)**0.8
    A_T1 = design_spectrum_A(T1)
    M_total = np.sum(m)
    Vb = M_total * A_T1 * g
    denominator = np.sum(m * h)
    F = Vb * (m * h) / denominator
    Mb = np.sum(F * h)
    print("\nQ4(d) Equivalent Static Lateral Force Procedure")
    print("===============================================")
    print(f"Fundamental period, T1 = {T1:.6f} s")
    print(f"Design spectral acceleration, A(T1) = {A_T1:.6f} g")
    print(f"\nTotal mass = {M_total:.3f} t")
    print(f"Base shear, Vb = {Vb:.2f} kN")
    print("\nVertical distribution of lateral forces")
    print("---------------------------------------")
    print("Floor | mass (t) | height (m) | F_i (kN)")
    for i in range(3): print(f"{i+1:>5d} | {m[i]:>8.3f} | {h[i]:>10.3f} | {F[i]:>8.2f}")
    print(f"\nCheck sum of floor forces = {np.sum(F):.2f} kN")
    print(f"Base overturning moment, Mb = {Mb:.2f} kNm")
    np.savetxt(data_dir / "Q4d_ESLFP_floor_forces.csv", np.column_stack((np.arange(1, 4), m, h, F)), delimiter=",", header="floor,mass_t,height_m,F_i_kN", comments="")
    with open(data_dir / "Q4d_ESLFP_summary.csv", "w") as f:
        f.write("quantity,value\n")
        for row in np.array([["T1_s", T1], ["A_T1_g", A_T1], ["Vb_kN", Vb], ["Mb_kNm", Mb]], dtype=object): f.write(f"{row[0]},{float(row[1]):.6f}\n")
    print("\nSaved:")
    print("Q4d_ESLFP_floor_forces.csv")
    print("Q4d_ESLFP_summary.csv")


def q4e_compare(script_dir: Path):
    import pandas as pd
    data_dir = script_dir / "data"
    figures_dir = script_dir / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rha_file   = data_dir / "Q3cdef_RHA_results.csv"
    rsa_u_file = data_dir / "Q4b_RSA_story_shears.csv"
    rsa_mb_file = data_dir / "Q4b_RSA_base_overturning_moment.csv"
    eslfp_file  = data_dir / "Q4d_ESLFP_summary.csv"
    rha_df   = pd.read_csv(rha_file)
    rsa_mb   = pd.read_csv(rsa_mb_file)
    eslfp_df = pd.read_csv(eslfp_file)
    rsa_v    = pd.read_csv(rsa_u_file)
    rha_vb = np.max(np.abs(rha_df["Vbase_kN"]))
    rha_mb_val = np.max(np.abs(rha_df["Mb_kNm"]))
# FIXED
    abssum_vb = float(rsa_v.loc[rsa_v["story"]==1, "V_ABSSUM_kN"].iloc[0])
    srss_vb   = float(rsa_v.loc[rsa_v["story"]==1, "V_SRSS_kN"].iloc[0])
    cqc_vb    = float(rsa_v.loc[rsa_v["story"]==1, "V_CQC_kN"].iloc[0])
    abssum_mb = float(rsa_mb.loc[rsa_mb["rule"]=="ABSSUM", "Mb_kNm"].iloc[0])
    srss_mb   = float(rsa_mb.loc[rsa_mb["rule"]=="SRSS",   "Mb_kNm"].iloc[0])
    cqc_mb    = float(rsa_mb.loc[rsa_mb["rule"]=="CQC",    "Mb_kNm"].iloc[0])
    eslfp_vb  = float(eslfp_df.loc[eslfp_df["quantity"]=="Vb_kN",   "value"].iloc[0])
    eslfp_mb  = float(eslfp_df.loc[eslfp_df["quantity"]=="Mb_kNm",  "value"].iloc[0])
    results = {
        "Method": ["RHA", "RSA ABSSUM", "RSA SRSS", "RSA CQC", "ESLFP"],
        "Base Shear Vb (kN)":              [rha_vb,  abssum_vb, srss_vb, cqc_vb, eslfp_vb],
        "Base Overturning Moment Mb (kNm)":[rha_mb_val, abssum_mb, srss_mb, cqc_mb, eslfp_mb]
}
    df = pd.DataFrame(results)
    rha_vb = df.loc[df["Method"] == "RHA", "Base Shear Vb (kN)"].iloc[0]
    rha_mb = df.loc[df["Method"] == "RHA", "Base Overturning Moment Mb (kNm)"].iloc[0]
    df["Vb Difference from RHA (%)"] = (df["Base Shear Vb (kN)"] - rha_vb) / rha_vb * 100
    df["Mb Difference from RHA (%)"] = (df["Base Overturning Moment Mb (kNm)"] - rha_mb) / rha_mb * 100
    df_rounded = df.round({"Base Shear Vb (kN)": 2, "Base Overturning Moment Mb (kNm)": 2, "Vb Difference from RHA (%)": 2, "Mb Difference from RHA (%)": 2})
    print("\nQ4(e) Comparison of RHA, RSA, and ESLFP")
    print("======================================")
    print(df_rounded.to_string(index=False))
    df_rounded.to_csv(data_dir / "Q4e_RHA_RSA_ESLFP_comparison.csv", index=False)
    try:
        latex_table = df_rounded.to_latex(index=False, caption="Comparison of RHA, RSA, and ESLFP base responses.", label="tab:rha_rsa_eslfp_comparison", float_format="%.2f")
    except Exception:
        latex_table = None
        print("Skipped LaTeX export for Q4(e) because the optional jinja2 dependency is not installed.")
    if latex_table is not None:
        with open(script_dir / "Q4e_RHA_RSA_ESLFP_comparison.tex", "w", encoding="utf-8") as f:
            f.write(latex_table)
    print("\nSaved:")
    print("Q4e_RHA_RSA_ESLFP_comparison.csv")
    if latex_table is not None:
        print("Q4e_RHA_RSA_ESLFP_comparison.tex")


if __name__ == "__main__":
    main()
