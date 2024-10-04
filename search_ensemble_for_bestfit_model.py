import numpy as np
import os
import seissolxdmf as sx
from scipy.signal import correlate
import argparse

parser = argparse.ArgumentParser(
    description="Search the best-fit model of the ensemble"
)

parser.add_argument("obs_path", help="path to observed waveforms")
parser.add_argument(
    "ensemble_dir",
    help="path to directory containing subdirectories with SeisSol output",
)
parser.add_argument(
    "--final_ind",
    nargs=1,
    metavar=("threshold"),
    default=([-1]),
    help="last index of the synthetic data to be loaded",
    type=int,
)
args = parser.parse_args()


def load_seissol_surface_receiver(filename, final_ind=-1):
    data = np.loadtxt(filename, skiprows=5, usecols=(0, 7, 8, 9)).T
    # output numpy array (4, ndt) containing time, v1, v2, v3
    return data[:, :final_ind]


def check_if_seissol_surface_receiver_file(filename):
    # currently only based on the naming convention, could add another test based on the file content
    out = False
    if "receiver" in filename.split("-"):
        out = True
    if filename.split(".")[-1] != "dat":
        out = False
    return out


def load_all_seissol_receivers_from_dir(path, final_ind=-1):
    files = os.listdir(path)
    surface_receiver = []
    for i in files:
        if check_if_seissol_surface_receiver_file(i):
            surface_receiver.append(i)

    # sorted list containing all surface receivers of the provided directory
    surface_receiver.sort()

    data = []
    for i in surface_receiver:
        data.append(load_seissol_surface_receiver(path + i, final_ind=final_ind))
    data = np.array(data)
    # output numpy array (number_of_receivers, 4, ndt)
    return data


def load_observations(path):
    # this is a prototype function that currently loads synthetic SeisSol data
    data = load_all_seissol_receivers_from_dir(path)
    # output numpy array (number_of_stations, 4, ndt)
    return data


def check_if_seissol_dir(path):
    # checks if the provided directory contains seissol surface receivers and a seissol surface xdmf
    out = False
    number_surface_receiver = 0
    files = os.listdir(path)

    for i in files:
        if i.split("-")[-1] == "surface.xdmf":
            try:
                seissolxdmf = sx.seissolxdmf(path + i)
                test = seissolxdmf.ReadNElements()
            except:
                print(path + i)
                print("A defect surface.xdmf file was detected.")
            else:
                out = True

    for i in files:
        if check_if_seissol_surface_receiver_file(i):
            number_surface_receiver += 1

    if number_surface_receiver == 0:
        out = False

    return out


def collect_seissol_dirs(path):
    # returns all directories containing SeisSol output
    dirs = []
    for i in os.listdir(path):
        try:
            if check_if_seissol_dir(path + i + "/"):
                dirs.append(i)
        except:
            continue

    dirs.sort()
    return dirs


def load_ensemble_waveforms(path, final_ind=-1):
    # loads synthetic data from every SeisSol output directory in path
    dirs = collect_seissol_dirs(path)
    print(f"Reading synthetic data from the following directories: {dirs}")
    data = []
    for i in dirs:
        data.append(
            load_all_seissol_receivers_from_dir(path + i + "/", final_ind=final_ind)
        )

    # output numpy array (number_of_models, number_of_stations, 4, ndt)
    return np.array(data), dirs


def correlate_and_shift_waveforms(obs_trace, syn_trace):
    # assumes that observed wavefroms are longer than the synthetics
    start_ind = np.argmax(correlate(obs_trace, syn_trace, mode="valid"))
    return obs_trace[start_ind : start_ind + syn_trace.shape[0]]


def calculate_rms_misfit(obs_trace, syn_trace):
    return np.sqrt(np.sum((obs_trace - syn_trace) ** 2) / syn_trace.shape[0])


def calculate_summed_model_misfit(obs_data, syn_data):
    misfit = 0
    for i in range(obs_data.shape[0]):
        for j in range(3):
            syn_trace = syn_data[i, j + 1]
            obs_trace = correlate_and_shift_waveforms(obs_data[i, j + 1], syn_trace)
            misfit += calculate_rms_misfit(obs_trace, syn_trace)

    return misfit


obs_data = load_observations(args.obs_path)
print(f"Observations contain {obs_data.shape[0]} stations.")
print(f"Observations contain {obs_data.shape[2]} time steps.")

syn_data, model_dir_names = load_ensemble_waveforms(
    args.ensemble_dir, final_ind=args.final_ind[0]
)
print(f"Synthetics contain {syn_data.shape[1]} stations.")
print(f"Synthetics contain {syn_data.shape[3]} time steps.")


model_misfits = np.zeros(len(model_dir_names))

for i in range(len(model_dir_names)):
    model_misfits[i] = calculate_summed_model_misfit(obs_data, syn_data[i])

print(f"List of all model misfits: {model_misfits}")

min_ind = np.argmin(model_misfits)
print(
    f"{model_dir_names[min_ind]} yields the smallest misfit ({model_misfits[min_ind]:.5f})."
)
print("Done.")
