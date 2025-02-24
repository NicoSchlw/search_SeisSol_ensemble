#!/bin/bash
obsdir='path/to/TABOO_waveforms'
ensemble_dir='path/to/catalog_scenario'
eq_time='2023,3,9,19,8,6.0'
python3 search_catalog_for_best_fit_model.py $eq_time $obsdir $ensemble_dir 
