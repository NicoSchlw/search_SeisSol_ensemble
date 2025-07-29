#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from pathlib import Path
import argparse
from scipy import interpolate
from scipy.signal import correlate
import numpy as np
import fnmatch
from pyproj import Transformer
from obspy import read
from obspy import read_inventory
from obspy import UTCDateTime

parser = argparse.ArgumentParser(
    description="Search the best-fit scenario of the catalog"
)

parser.add_argument(
    "eq_time",
    help="Date and time of the earthquake. String with the following structure: YYYY,MM,DD,HH,MM,SS.S. E.g: '2023,3,9,19,8,6.0'",
)

parser.add_argument(
    "obsdir",
    help="name of the folder in which the waveforms are stored",
)

parser.add_argument(
    "ensemble_dir",
    help="name of the folder in which the scenarios are stored",
)

parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed infos")

args = parser.parse_args()


def extractCoordStation(obsdir):
    """Look in the metadata file in obsdir for the station coordinates"""     
    for file in os.listdir(obsdir):
        if "metadata" in file.lower():
            metadata_file = os.path.join(obsdir, file)
            break      
    # Read the metadata file
    inventory = read_inventory(metadata_file)
    # Extract station coordinates
    station_coords = {}
    for network in inventory:
        for station in network:
            station_coords[station.code] = (station.longitude, station.latitude)
    return station_coords


def readSeisSolReceiver(file):
    """READ SeisSol receiver"""
    with open(file) as fid:
        fid.readline()
        variablelist = fid.readline()[11:].split(",")
        variablelist = np.array([a.strip().strip('"') for a in variablelist])
        xsyn = float(fid.readline().split()[2])
        ysyn = float(fid.readline().split()[2])
        zsyn = float(fid.readline().split()[2])
        synth = np.loadtxt(fid)
        # Extract only relevant columns
        selected_vars = ["Time", "v1", "v2", "v3"]
        indices = [i for i, var in enumerate(variablelist) if var in selected_vars]
        synth = synth[:, indices]  # Select columns matching "time", "v1", "v2", "v3"      
    return ([xsyn, ysyn, zsyn], synth)


def readObservation(obsdir,station):
    """READ mseed file"""
    # Look for mseed file for the given station in directory obsdir
    file_obs = []
    for file in os.listdir(obsdir):
        if fnmatch.fnmatch(file, f"*{station}*.mseed"):
            file_obs = file
           # print(f"Found mseed file for station {sta2comp} in directory {obsdir}") 
        
    if len(file_obs) == 0:
        if args.verbose:
            print(f"mseed file not found for station {station} in directory {obsdir}")
        obs_trace = 0
    else:
        obs_trace = read(f"{obsdir}/{file_obs}", format="MSEED")
        # Look in priority for BH channels -> if not available, look for HH -> if not EH
        channel_priority = ["BH*", "HH*", "EH*"]
        for channel_pattern in channel_priority:
            obs_filtered = obs_trace.select(channel=channel_pattern)
            if len(obs_filtered) > 0:
                obs_trace = obs_filtered
                if args.verbose:
                    print(f"Extracting channels {channel_pattern}")
                break  # Stop searching once we find a valid channel group
        # Apply bandpass filtering       
        obs_trace.filter(
               "bandpass", freqmin=0.005, freqmax=1.0, corners=2, zerophase=True
               ) 
    return obs_trace


def matchStation2Receiver(receiver_coords,station_coords):
    """"Retrieve the station corresponding to a given SeisSol receiver"""   
    transformer = Transformer.from_crs("epsg:32633", "epsg:4326", always_xy=True)
    lon,lat,depth = transformer.transform(receiver_coords[0], receiver_coords[1], receiver_coords[2])   
    sta2comp = []
    for station, coordstat in station_coords.items():
        if (abs(lon - coordstat[0]) < 2e-3) & (abs(lat - coordstat[1]) < 2e-3):
            #print(f"Found matching station: {station}")      
            sta2comp = station
    return sta2comp


def computeMisfit(obs_trace,synth,eq_time): 
    misfit_comp=0
    component=['E','N','Z']    
    for comp  in range(3):
        time_obs = obs_trace.select(component=component[comp])[0].times(reftime=eq_time)
        data_obs = obs_trace.select(component=component[comp])[0].data   
        time_synth = synth[:,0]
        data_synth = synth[:,1 + comp]
        
        min_duration = min(time_obs[-1],time_synth[-1])
        data_synth = data_synth[np.where(time_synth<=min_duration)]
        time_synth = time_synth[np.where(time_synth<=min_duration)]        
        f = interpolate.interp1d(time_obs,data_obs)
        data_obs_interp = f(time_synth)
        
        #data_obs_shift = correlate_and_shift_waveforms(data_obs, data_synth)
        misfit_comp +=  calculate_rms_misfit(data_obs_interp, data_synth)
    return misfit_comp
     
   
def correlate_and_shift_waveforms(obs_trace, syn_trace):
    # assumes that observed wavefroms are longer than the synthetics
    start_ind = np.argmax(correlate(obs_trace, syn_trace, mode="valid"))
    return obs_trace[start_ind : start_ind + syn_trace.shape[0]]


def calculate_rms_misfit(obs_trace, syn_trace):
    return np.sqrt(np.sum((obs_trace - syn_trace) ** 2) / syn_trace.shape[0])


def saveMisfit(output_file,dic,header):    
    with open(output_file, "w") as f:
        f.write(f"{header}\n")
        for key, value in dic.items():
            f.write(f"{key} {value:.10f}\n")  # Format with 6 decimal places


eq_time = [int(x) if x.isdigit() else float(x) for x in args.eq_time.split(',')]
eq_time = UTCDateTime(eq_time[0], eq_time[1], eq_time[2], eq_time[3], eq_time[4], eq_time[5])

# Extract station coordinates in xml metadata file
station_coords = extractCoordStation(args.obsdir)
nbWaveform=len(list(Path(args.obsdir).glob("*.mseed")))
print(f"{len(station_coords)} stations listed in metadata file")
print(f"{nbWaveform} waveforms in {args.obsdir}")
if (len(station_coords)) > nbWaveform:
    diff =  (len(station_coords)) - len(list(Path(args.obsdir).glob("*.mseed")))
    print(f"Warning: missing mseed files for {diff} stations listed in metadata file\n")
    
# Retrieve the scenarios available in the catalog
if os.path.exists(args.ensemble_dir) and os.path.isdir(args.ensemble_dir):
    model_dir_names = [d for d in os.listdir(args.ensemble_dir) if os.path.isdir(os.path.join(args.ensemble_dir, d))]
    if 'inputs' in model_dir_names:
        model_dir_names.remove('inputs')
    model_dir_names = sorted(model_dir_names, key=int)
    print(f"Found {len(model_dir_names)} scenarios in Alto Tiberina catalog\n")
else:
    print(f"Error: The directory '{args.ensemble_dir}' does not exist.")

    
model_misfits =  {} 
# Loop on each scenario
for i in range(len(model_dir_names)):    
    print(f"Computing misfit for scenario {model_dir_names[i]}")
    
    # Retrieve the receiver output files
    files = [f for f in os.listdir(args.ensemble_dir + "/" + model_dir_names[i]) if "receiver" in f]
    station_misfits = {}
    
    # Loop on each receiver file
    for j in files:
        # Read SeisSol receiver
        coords, synth = readSeisSolReceiver(args.ensemble_dir + "/" + model_dir_names[i] + "/" + j)
        
        # Find the corresponding station
        sta2comp = matchStation2Receiver(coords, station_coords)
        if len(sta2comp) == 0:
            if args.verbose:
                print(f"Station not found in the metadata file for receiver {j}")
            continue
        else:
            if args.verbose:
                print(f"Found matching station for receiver {j}: {sta2comp}")
            
        # Load waveforms of the corresponding station
        obs_trace = readObservation(args.obsdir, sta2comp)
        if type(obs_trace) == int:
            continue

        # Compute station misfit
        station_misfits[sta2comp] = computeMisfit(obs_trace, synth, eq_time)
        
    print(f"Misfit computed for {len(station_misfits)} receivers over a total of {len(files)}\n")
    
    # Save text file with misfit values per station 
    saveMisfit(f"scenario_{model_dir_names[i]}_station_misfit.txt", station_misfits, "station misfit")
    model_misfits[model_dir_names[i]] =  sum(station_misfits.values())  

# Save text file with misfit values per scenario 
saveMisfit("scenario_misfit.txt", model_misfits, "scenario misfit")

# Find scenario yielding the lowest misfit
best_model = min(model_misfits, key=model_misfits.get)
print(
    f"\nScenario {best_model} yields the smallest misfit ({model_misfits[best_model]:.5f}).\n"
)
print("Done.")
