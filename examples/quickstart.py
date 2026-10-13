"""
Quickstart example: run the default simulation and plot the population
of each spin component as a function of time.

    pixi run example
or
    python examples/quickstart.py

The result is cached in data/output/, so a second run only re-plots.
Pass --recompute to force a new simulation (~1 min).
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from discrete_states_code.run_simulation import Parameters, run

ROOT = Path(__file__).resolve().parent.parent


def main():
    p = Parameters()   # edit the parameters here, e.g. Parameters(k_modes=7)
    df = run(p, data_dir=str(ROOT / 'data'),
             recompute='--recompute' in sys.argv)

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor='#fcfcfb')
    ax.set_facecolor('#fcfcfb')
    ax.plot(df['time_ms'], df['N0'], color='#2a78d6', lw=2,
            label=r'$N_0$ (spin $\uparrow$)')
    ax.plot(df['time_ms'], df['N1'], color='#eb6834', lw=2,
            label=r'$N_1$ (spin $\downarrow$)')

    ax.set_xlabel('time [ms]', color='#52514e')
    ax.set_ylabel('population fraction', color='#52514e')
    ax.set_title(f'Spin populations, {p.k_modes} momentum modes, '
                 rf'$\Omega_R$={p.omega_r} $E_r$, '
                 rf'$\delta$: {p.delta_i:g}$\to${p.delta_f:g} $E_r$',
                 color='#0b0b0b', loc='left')
    ax.set_xlim(df['time_ms'].min(), df['time_ms'].max())
    ax.set_ylim(0, 1)
    ax.grid(axis='y', color='#e1e0d9', lw=0.8)
    ax.tick_params(colors='#898781')
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color('#c3c2b7')
    ax.legend(frameon=False, loc='center right', labelcolor='#0b0b0b')
    fig.tight_layout()

    out = ROOT / 'examples' / 'quickstart.png'
    fig.savefig(out, dpi=150)
    print(f'figure saved to {out}')
    plt.show()


if __name__ == '__main__':
    main()
