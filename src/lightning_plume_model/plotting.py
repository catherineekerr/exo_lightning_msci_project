from constants import SimulationParameters
from constants import PhysicalConstants
from pathlib import Path
from typing import Tuple, Union
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

import os
import ast

from lightning_work import sim_params_container
from lightning_work import results

PROJECT_NAME = "convective_plume_earth"
CONST = PhysicalConstants()




def plot_comparison(
    results: dict,
    sim_params_container: dict[SimulationParameters],
    output_dir: Union[str, Path],
):
    """Create comparison plots for a series of simulations."""

    @dataclass
    class _PlotConfig:
        """Configuration for a single plot."""

        ylabel: str
        title: str
        units: str

    plot_configs = {
        "velocity": _PlotConfig(
            ylabel="Vertical velocity",
            title="Vertical Plume Velocity",
            units="m s-1",
        ),
        "plume_temp": _PlotConfig(
            ylabel="Temperature", title="Plume Temperature", units="K"
        ),
        "env_temp": _PlotConfig(
            ylabel="Temperature", title="Environment Temperature", units="K"
        ),
        "temp_diff": _PlotConfig(
            ylabel="Temperature difference",
            title="Plume-Environment Temperature Difference",
            units="K",
        ),
        "plume_radius": _PlotConfig(ylabel="Radius", title="Plume Radius", units="m"),
        "flash_rate": _PlotConfig(
            ylabel="Flash rate",
            title="Lightning Flash Rate",
            units="flashes s-1 km-2",
        ),
    }

    # Create figure with mosaic layout using plot_configs keys
    mosaic = [list(plot_configs.keys())[:3], list(plot_configs.keys())[3:]]

    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    axes = fig.subplot_mosaic(mosaic)

    # Plot each variable
    handles, labels = None, None
    for name, config in plot_configs.items():
        ax = axes[name]

        for run_label, data in results.items():
            # Convert pressure to bar
            if name == "flash_rate":  # Flash rate uses fewer points
                y = (
                    data["pressure"][
                        :: sim_params_container[run_label].flash_rate_sampling
                    ]
                    * CONST.pa_to_bar
                )
            else:
                y = data["pressure"] * CONST.pa_to_bar

            if name == "temp_diff":
                x = data["plume_temp"] - data["env_temp"]  # Plume temp - Env temp
            else:
                x = data[name]

            line = ax.plot(x, y, label=run_label, linewidth=1.5)

            # Capture handles and labels from the first subplot
            if handles is None:
                handles, labels = [], []
            if name == list(plot_configs.keys())[0]:
                handles.extend(line)
                labels.append(run_label)

        ax.set_ylabel("Pressure [bar]")
        ax.set_ylim(sim_params_container[run_label].start_pressure * CONST.pa_to_bar, 0)
        ax.set_xlabel(f"{config.ylabel} [{config.units}]")
        ax.set_title(config.title)
        ax.grid(True, alpha=0.3)

    # Add a single legend for the entire figure
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(labels),
        frameon=True,
        fontsize=10,
    )

    fig.suptitle('1D Model of Convective Plume for Earth', fontsize="x-large", fontweight="bold", y=1.075)

    filename = f"{PROJECT_NAME}.png"
    fig.savefig(Path(output_dir) / filename, dpi=150, bbox_inches="tight")
    print(f"  Saved: {filename}")
    plt.close()


file_name = '280__20__0p8__0p0.txt'


script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, file_name)


with open(file_path) as file:
    data = file.read()

arrays = eval(data, {"array": np.array})

pressure = arrays['pressure']
print(pressure)


flash_rate = arrays['flash_rate']
print(flash_rate)

print("Generating plots...")
outdir = Path(__file__).parent / "output"
outdir.mkdir(exist_ok=True, parents=True)
plot_comparison(results, sim_params_container, outdir)
print("Done.")