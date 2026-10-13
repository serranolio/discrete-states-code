# -*- coding: utf-8 -*-
"""
discrete momentum modes simulation
geometry: NA
system: 2-component BEC
author: Federico Serrano
Physics and Astronomy Department
Washington State University

Usage (from the command line):
    discrete-states --help
    python -m discrete_states_code.run_simulation --help
"""
import argparse
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from numpy import pi
from scipy.integrate import solve_ivp


"""
1. global parameters ---------------------------------------------------
"""

@dataclass
class Parameters:
    """Simulation parameters.

    Energies are in units of the recoil energy E_r = h * f_recoil and
    momenta in units of the recoil momentum k_r.
    """
    f_recoil: float = 1960.0          # recoil frequency [Hz]
    gn: float = 0.6567617303620249    # mean-field interaction energy [E_r]
    k_modes: int = 5                  # number of discrete momentum modes

    omega_l_i: float = 0.2            # initial lattice depth [E_r]
    omega_l_f: float = 0.2            # final lattice depth [E_r]

    delta_i: float = 1.0              # initial Raman detuning [E_r]
    delta_f: float = 0.0              # final Raman detuning [E_r]

    omega_r: float = 2.7              # Raman coupling strength [E_r]

    ramp_time_ms: float = 0.01        # linear ramp duration [ms]
    simulation_time_ms: float = 40.0  # total evolution time [ms]
    frames: int = 1000                # number of saved time steps

    @property
    def k_l(self):
        """Lattice wave vector [k_r]."""
        return np.sqrt(1 - self.omega_r**2/16)

    @property
    def ramp_time(self):
        """Ramp duration in dimensionless units (1 / (2 pi f_recoil))."""
        return self.ramp_time_ms*1e-3 * (2*pi*self.f_recoil)

    @property
    def simulation_time(self):
        """Simulation time in dimensionless units (1 / (2 pi f_recoil))."""
        return self.simulation_time_ms*1e-3 * (2*pi*self.f_recoil)


"""
2. output format -------------------------------------------------------
"""

def output_file_name(p, data_dir='data'):
    params = {'omega_l_i': p.omega_l_i,
              'omega_l_f': p.omega_l_f,
              'delta_i': p.delta_i,
              'delta_f': p.delta_f,
              'ramp_time': p.ramp_time_ms,
              'kL': p.k_l,
              'gn': p.gn,
              'modes': p.k_modes}
    params_str = ''
    for keys, values in params.items():
        params_str += (keys + '_%.3f_' % values)
    return os.path.join(data_dir, 'output',
                        'polarization_' + params_str + '.csv')


def input_file_name(p, data_dir='data'):
    return os.path.join(data_dir, 'input',
                        f'psi_ground_state_omega_l_{p.omega_l_i:.3f}_'
                        + f'delta_{p.delta_i:.3f}_'
                        + f'omega_r_{p.omega_r:.3f}_'
                        + f'kL_{p.k_l:.3f}_'
                        + f'gn_{p.gn:.3f}_'
                        + f'modes_{p.k_modes:.3f}.npy')


"""
3. utility functions ---------------------------------------------------
"""

# single-particle hamiltonian
def H_0(p, k, delta_fun, omega_l_fun, t):
    delta = delta_fun(t)
    omega_l = omega_l_fun(t)
    k_modes = p.k_modes

    sigma_z = np.array([[1.0, 0.0],
                        [0.0, -1.0]])

    q = k + 2*p.k_l*(np.arange(k_modes) - (k_modes - 1) / 2)

    detuning = -delta/2 * sigma_z

    raman_coupling = np.array([[0.0, p.omega_r / 2],
                               [p.omega_r / 2, 0.0]])
    lattice = -2.0 * omega_l * (np.eye(k_modes, k=1) / 4.0
                                + np.eye(k_modes, k=-1) / 4.0)
    T = (np.kron(np.eye(2), np.diag(q**2))
         - 2.0 * np.kron(sigma_z, np.diag(q))
         + np.kron(sigma_z**2, np.eye(k_modes)))
    return (T + np.kron(detuning + raman_coupling, np.eye(k_modes))
            + np.kron(np.eye(2), lattice))

# interaction hamiltonian
def H_I(p, psi):

    dim_s = 2
    dim_k = len(psi) // dim_s

    I = np.zeros((dim_k, dim_k), dtype='complex')

    for i in range(dim_k):
        c = np.trace(np.outer(psi.conj(), psi)
                     * np.kron(np.eye(dim_s), np.ones((dim_k, dim_k))),
                     offset=i)
        I += c * np.eye(dim_k, k=i, dtype='complex')

    I = I + I.conj().T - np.eye(dim_k) * np.diagonal(I.conj())
    return p.gn * np.kron(np.eye(dim_s), I.conj())

# get spectrum function
def get_spectrum(p, k_):
    def delta_fun(t):
        return p.delta_i

    def omega_l_fun(t):
        return p.omega_l_i

    def i_dpsi_dt(t, psi):
        H_psi = (H_0(p, k_, delta_fun, omega_l_fun, t) + H_I(p, psi)) @ psi
        chemical_potential = (psi.conj().T @ H_psi) / (psi.conj().T @ psi)
        return -H_psi + chemical_potential * psi

    psi0 = np.ones(p.k_modes*2) + 0j
    psi0 = psi0 / (psi0.conj().T @ psi0)

    # imaginary time evolution
    psi_gs = solve_ivp(fun=i_dpsi_dt,
                       t_span=[0, 100],
                       y0=psi0,
                       t_eval=[100],
                       atol=1e-9,
                       rtol=1e-9).y
    psi_gs.shape = psi0.shape

    return (psi_gs.conj().T @
            ((H_0(p, k_, delta_fun, omega_l_fun, 0) + H_I(p, psi_gs)/2) @
              psi_gs)).real

# get value of k at band-minima
def get_k_min(p):
    spectrum = np.zeros(50)
    k_ = np.linspace(-p.k_l, p.k_l, 50)
    for i in range(50):
        spectrum[i] = get_spectrum(p, k_[i])
    return k_[np.argmin(spectrum)]

# get ground_state
def get_psi_gs(p, k_min):
    def delta_fun(t):
        return p.delta_i

    def omega_l_fun(t):
        return p.omega_l_i

    def i_dpsi_dt(t, psi):
        H_psi = (H_0(p, k_min, delta_fun, omega_l_fun, t) + H_I(p, psi)) @ psi
        chemical_potential = (psi.conj().T @ H_psi) / (psi.conj().T @ psi)
        return -H_psi + chemical_potential * psi

    psi0 = np.ones(p.k_modes*2) + 0j
    psi0 = psi0 / np.sqrt(psi0.conj().T @ psi0)
    psi_gs = solve_ivp(fun=i_dpsi_dt,
                       t_span=[0, 500],
                       y0=psi0,
                       t_eval=[500],
                       atol=1e-10,
                       rtol=1e-10).y
    psi_gs.shape = psi0.shape

    return psi_gs

# simulation sequence
def simulation_sequence(p, delta_fun, omega_l_fun, k_min, psi_initial):

    def dpsi_dt(t, psi):
        H_psi = (H_0(p, k_min, delta_fun, omega_l_fun, t) + H_I(p, psi)) @ psi
        return -1j * H_psi
    t_axis = np.arange(p.frames)/(p.frames-1) * p.simulation_time
    t_axis_ms = np.arange(p.frames)/(p.frames-1) * p.simulation_time_ms
    sol = solve_ivp(fun=dpsi_dt,
                    y0=psi_initial,
                    t_span=[0, p.simulation_time],
                    t_eval=t_axis,
                    atol=1e-9,
                    rtol=1e-9)
    y = np.array(sol.y)
    y.shape = (2, p.k_modes, p.frames)
    y0 = y[0, :, :]
    y1 = y[1, :, :]

    n0 = (y0.conj()*y0).sum(axis=0).real
    n1 = (y1.conj()*y1).sum(axis=0).real

    return (t_axis_ms, n0, n1)

# lmg simulation
def lmg_simulation(p, delta_fun, omega_l_fun, chi, Sz0, theta0):
    state0 = [Sz0, theta0]
    t_axis = np.arange(p.frames)/(p.frames-1) * p.simulation_time
    t_axis_ms = np.arange(p.frames)/(p.frames-1) * p.simulation_time_ms

    def lmg_eom(t, state):
        delta = delta_fun(t)
        omega_l = omega_l_fun(t)

        Sz = state[0]
        theta = state[1]

        dSz_dt = -omega_l * chi *  np.sqrt(1-Sz**2) * np.sin(theta)
        dtheta_dt = (-p.gn * chi**2 * Sz
                     + omega_l * chi * Sz / np.sqrt(1 - Sz**2) * np.cos(theta)
                     + delta)
        return np.array([dSz_dt, dtheta_dt])

    sol = solve_ivp(fun=lmg_eom,
                    y0=state0,
                    t_span=[0, p.simulation_time],
                    t_eval=t_axis,
                    atol=1e-9,
                    rtol=1e-9)

    return (t_axis_ms, sol.y[0, :])

def make_ramps(p):
    """Linear ramps of detuning and lattice depth over p.ramp_time."""
    def delta_ramp(t):
        delta = ((p.delta_i + (p.delta_f - p.delta_i) * t/p.ramp_time)
                 * (t <= p.ramp_time)
                 + p.delta_f * (t > p.ramp_time))
        return delta

    def omega_l_ramp(t):
        omega_l = ((p.omega_l_i + (p.omega_l_f - p.omega_l_i) * t/p.ramp_time)
                   * (t <= p.ramp_time)
                   + p.omega_l_f * (t > p.ramp_time))
        return omega_l

    return delta_ramp, omega_l_ramp

def print_model_parameters(p, k0, delta_ramp, omega_l_ramp):
    h0 = H_0(p, k0, delta_ramp, omega_l_ramp, 0)
    assert np.allclose(h0, h0.T.conj())

    e, v = np.linalg.eigh(h0)

    k_modes = p.k_modes
    p_left = np.zeros(k_modes*2, dtype='complex')
    p_right = np.zeros(k_modes*2, dtype='complex')

    p_left[int((k_modes-1) // 2)] = 1.0
    p_left[int((k_modes-1) // 2 + k_modes)] = 1.0

    p_right[int((k_modes-1) // 2 + 1)] = 1.0
    p_right[int((k_modes-1) // 2 + k_modes + 1)] = 1.0

    chi_left = v[:, 0][[int((k_modes-1)//2),
                        int((k_modes-1)//2 + k_modes)]]
    chi_right = v[:, 0][[int((k_modes-1)//2 + 1),
                         int((k_modes-1)//2 + k_modes + 1)]]

    e_l = (v[:, 0] * p_left).conj().T @ (h0 @ (v[:, 0] * p_left))
    e_r = (v[:, 0] * p_right).conj().T @ (h0 @ (v[:, 0] * p_right))
    print('left/right overlap: chi=%.6f' % (chi_left @ chi_right))
    print('Model detuning: DE=%.6f' %(e_r - e_l).real)


"""
4. execute simulation --------------------------------------------------
"""

def run(p=None, data_dir='data', recompute=False):
    """Run the simulation and return a DataFrame with columns
    time_ms, N0, N1 (population of each spin component).

    Results are cached in `data_dir`/output and the ground state in
    `data_dir`/input; an existing result is loaded instead of recomputed
    unless `recompute` is True.
    """
    if p is None:
        p = Parameters()
    os.makedirs(os.path.join(data_dir, 'output'), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'input'), exist_ok=True)
    out_file = output_file_name(p, data_dir)
    in_file = input_file_name(p, data_dir)

    if not recompute and os.path.isfile(out_file):
        print("opening data file --> done")
        print("simulation already exists: %s" % out_file)
        return pd.read_csv(out_file)

    print("simulation sequence --> start")
    delta_ramp, omega_l_ramp = make_ramps(p)
    print("computing band-minima")
    k0 = get_k_min(p)
    print("computing band-minima --> k_min = %.5f" % k0)
    if not recompute and os.path.isfile(in_file):
        print("loading ground state")
        psi_gs = np.load(in_file)
    else:
        print("computing ground state")
        psi_gs = get_psi_gs(p, k0)
        np.save(in_file, psi_gs)
    t, n0, n1 = simulation_sequence(p, delta_ramp, omega_l_ramp, k0, psi_gs)
    print("simulation sequence --> done")
    df = pd.DataFrame()
    df['time_ms'] = t
    df['N0'] = n0
    df['N1'] = n1
    df.to_csv(out_file, index=False)
    print("results saved to %s" % out_file)

    print_model_parameters(p, k0, delta_ramp, omega_l_ramp)
    return df


def main(argv=None):
    defaults = Parameters()
    parser = argparse.ArgumentParser(
        description="Discrete momentum-mode simulation of a spin-orbit "
                    "coupled two-component BEC in an optical lattice.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    for name, value in asdict(defaults).items():
        parser.add_argument('--' + name.replace('_', '-'), dest=name,
                            type=type(value), default=value)
    parser.add_argument('--data-dir', default='data',
                        help="directory where input/ and output/ are stored")
    parser.add_argument('--recompute', action='store_true',
                        help="ignore cached results and run again")
    args = vars(parser.parse_args(argv))
    data_dir = args.pop('data_dir')
    recompute = args.pop('recompute')
    run(Parameters(**args), data_dir=data_dir, recompute=recompute)


if __name__ == "__main__":
    main()
