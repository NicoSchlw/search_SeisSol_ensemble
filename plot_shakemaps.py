import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seissolxdmf
import matplotlib.ticker as ticker
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import argparse

parser = argparse.ArgumentParser(
    description="generate shake map figure from SeisSol output"
)
parser.add_argument(
    "filename",
    help="output from ComputeGroundMotionParametersFromSurfaceOutput_Hybrid.py",
)
args = parser.parse_args()

# Script that calculates ground motion parameters from SeisSol output:
# https://github.com/SeisSol/SeisSol/blob/master/postprocessing/science/GroundMotionParametersMaps/ComputeGroundMotionParametersFromSurfaceOutput_Hybrid.py


def get_xyz_from_connect(geom, connect):
    # Genrate array with coordinates of triangle midpoints (same dimension as connect)
    xyz = np.zeros_like(connect, dtype=float)
    xyz = (1.0 / 3.0) * (
        geom[connect[:, 0], :] + geom[connect[:, 1], :] + geom[connect[:, 2], :]
    )
    return xyz


def get_cbar_level(data, major_ticks):
    min_value = major_ticks[np.argmin((np.percentile(data, 4) - major_ticks) ** 2)]
    max_value = major_ticks[np.argmin((np.percentile(data, 96) - major_ticks) ** 2)]
    return np.geomspace(min_value, max_value, 14)


def custom_terrain_cmap(z):
    min_value = np.round(np.max((np.min(z), -800)), -2)
    max_value = np.round((np.min((np.max(z), 4500))), -2)
    step_size = np.round((max_value - min_value) / 13, -2)
    cbar_levels = np.arange(min_value, max_value, step_size)
    if cbar_levels[0] < 0.0:
        cbar_levels -= cbar_levels[np.argmin(np.abs(cbar_levels))]
    cmap_min_index = np.max((50 + int(cbar_levels[0] / 20), 10))
    # cmap_max_index=np.min((50+int(cbar_levels[-1]/23), 240))
    cmap_max_index = np.min((50 + int(180 * np.sqrt(cbar_levels[-1] / 4500)), 230))
    cmap_new = mpl.cm.terrain(np.arange(256))[cmap_min_index:cmap_max_index]
    cmap_new = mpl.colors.ListedColormap(
        cmap_new, name="custom_terrain1", N=cmap_new.shape[0]
    )
    return cmap_new, cbar_levels


def plot_shakemaps():

    filename = args.filename

    # increase to decrease the resolution and speed up the figure generation (useful when modifying/debugging)
    step = 1

    cmaps = ["OrRd", "YlOrRd", "Spectral_r", "turbo"]
    labels = ["PGD (m)", "PGV (m/s)", "PGA (m/s$^2$)", "Altitude (m)"]

    sx = seissolxdmf.seissolxdmf(filename)
    variables = sx.ReadAvailableDataFields()
    print(f"The provided xdmf file contains the following variables: {variables}")

    xyz = get_xyz_from_connect(sx.ReadGeometry(), sx.ReadConnect())
    print(f"Number of surface receivers: {xyz.shape[0]}")

    x = xyz[:, 0] * 1e-3
    y = xyz[:, 1] * 1e-3
    z = xyz[:, 2]

    # list containing the 3 fields that will be plotted
    data = [sx.ReadData("PGD"), sx.ReadData("PGV"), sx.ReadData("PGA")]
    data = [np.squeeze(data[i]) for i in range(len(data))]

    # hardcoded values covering the relevant range of seismic hazard assessment
    minor_ticks = np.unique(
        np.concatenate([np.linspace(1, 10, 10) * i for i in [0.001, 0.01, 0.1, 1, 10]])
    )
    major_ticks = np.unique(
        np.concatenate([np.array((1, 2, 5, 10)) * i for i in [0.001, 0.01, 0.1, 1, 10]])
    )

    cbar_levels = [get_cbar_level(data[i], major_ticks) for i in range(3)]

    # remove edges that may be affected from the projection
    xlim = [
        np.min(x) + (np.max(x) - np.min(x)) * 0.03,
        np.max(x) - (np.max(x) - np.min(x)) * 0.03,
    ]
    ylim = [
        np.min(y) + (np.max(y) - np.min(y)) * 0.03,
        np.max(y) - (np.max(y) - np.min(y)) * 0.03,
    ]

    fig, ax = plt.subplots(2, 2, figsize=(10, 8))

    # plot shake maps
    for i in range(len(data)):
        tri = ax[int(i / 2), i % 2].tricontourf(
            x[::step],
            y[::step],
            data[i][::step],
            cmap=cmaps[1],
            extend="both",
            norm=colors.LogNorm(vmin=cbar_levels[i][0], vmax=cbar_levels[i][-1]),
            levels=cbar_levels[i],
        )
        cbar = plt.colorbar(tri, ax=ax[int(i / 2), i % 2])

        major_ticks_tmp = major_ticks[
            (major_ticks <= cbar_levels[i][-1]) & (major_ticks >= cbar_levels[i][0])
        ]
        minor_ticks_tmp = minor_ticks[
            (minor_ticks <= cbar_levels[i][-1]) & (minor_ticks >= cbar_levels[i][0])
        ]
        cbar.set_ticks(major_ticks_tmp)
        cbar.set_ticks(minor_ticks_tmp, minor=True)

    # plot topography
    try:
        cmap_topo, cbar_levels_topo = custom_terrain_cmap(z)
        tri = ax[1, 1].tricontourf(
            x[::step],
            y[::step],
            z[::step],
            cmap=cmap_topo,
            extend="both",
            levels=cbar_levels_topo,
        )

    except:
        print(
            "Custom terrain colormap did not work, falling back to the standard colormap."
        )
        tri = ax[1, 1].tricontourf(
            x[::step], y[::step], z[::step], cmap="terrain", extend="both"
        )

    cbar = plt.colorbar(tri, ax=ax[1, 1])

    for i in np.ndenumerate(ax):
        ax[i[0]].set_title(labels[i[0][0] * 2 + i[0][1]], pad=4)
        ax[i[0]].set_ylim(ylim[0], ylim[1])
        ax[i[0]].set_xlim(xlim[0], xlim[1])
        ax[i[0]].set_aspect("equal", adjustable="box")
        ax[i[0]].tick_params(direction="in", right=True, top=True)
        ax[i[0]].xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[i[0]].yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax[i[0]].tick_params(axis="both", labelsize=8)
        ax[i[0]].set_xlabel("x (km)", size=8, labelpad=1)
        ax[i[0]].set_ylabel("y (km)", size=8, labelpad=2)

    plt.savefig(
        "shakemaps.jpg",
        dpi=300,
        facecolor="white",
        transparent=False,
        bbox_inches="tight",
    )
    plt.close()
    print("Plotting done.")


plot_shakemaps()
