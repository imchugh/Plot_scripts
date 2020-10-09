#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 14:45:41 2017

@author: ian
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import pdb

def plot_ustar(path_to_file, num_cats = 30, vars_dict = None,
               ustar_threshold = None, light_threshold = 10):

    """
    Plot CO2 flux as a functions of friction velocity
    Args:
        * path_to_file (str) - path to netcdf file containing requisite
          variables (CO2 flux, friction velocity, insolation)
    Kwargs:
        * num_cats (int) - number of categories to split ustar into
        * vars_dict (dict) - dictionary containing mapping of variables to
          dataset-specific variable names (see _define_default_internal_names()
          function below for key names); if not user-specified, default names
          will be used, but will fail if these names do not appear in the data
        * ustar_threshold (float) - can be used to impose a vertical line at
          the approximate threshold defined by user; not used by default
        * light_threshold (int or float) - light level used to define day /
          night
    """

    # Open, dump data into ds and convert to df
    df = _make_df(path_to_file, vars_dict)

    # Group by quantile and generate mean
    noct_df = df.loc[df.Fsd < light_threshold].copy()
    noct_df.drop('Fsd', axis=1, inplace=True)
    noct_df['ustar_cat'] = pd.qcut(df.ustar, num_cats,
                                   labels = np.linspace(1, num_cats, num_cats))
    means_df = noct_df.dropna().groupby('ustar_cat').mean()

    # Plot
    fig, ax1 = plt.subplots(1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax1.set_ylabel(r'$R_e\/(\mu mol\/m^{-2}\/s^{-1})$', fontsize = 18)
    ax1.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 18)
    ax1.tick_params(axis = 'x', labelsize = 14)
    ax1.tick_params(axis = 'y', labelsize = 14)
    ax1.yaxis.set_ticks_position('left')
    ax1.xaxis.set_ticks_position('bottom')
    if ustar_threshold: ax1.axvline(ustar_threshold, color='black', lw=0.5)
    ax1.plot(means_df.ustar, means_df.Fc, marker='o', mfc='None',
            color = 'black', ls=':', label='Turbulent flux')
    if 'Fc_storage' in noct_df.columns:
        ax1.plot(means_df.ustar, means_df.Fc_storage, marker='s', mfc='None',
                color = 'black', ls='-.', label='Storage')
        ax1.plot(means_df.ustar, means_df.Fc + means_df.Fc_storage,
                marker = '^', mfc = '0.5', color = '0.5', label='Apparent NEE')
        ax1.legend(frameon=False)
    ax2 = ax1.twinx()
    ax2.set_ylim([10,20])
    ax2.tick_params(axis = 'y', labelsize = 14)
    ax2.set_ylabel(r'$Temperature\/(^oC)$', fontsize = 18)
    ax2.plot(means_df.ustar, means_df.Ta)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_internal_names():

    return {'flux_name': 'Fc',
            'storage_name': 'Fc_storage',
            'insolation_name': 'Fsd',
            'friction_velocity_name': 'ustar',
            'temperature_name': 'Ta'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _make_df(path_to_file, external_names):

    # Open the data
    ds = xr.open_dataset(path_to_file)
    ds = ds.sel(latitude=ds.latitude[0], longitude=ds.longitude[0], drop=True)
    df = ds.to_dataframe()
    ds.close()
    df.index = df.index.drop_duplicates()
    df.replace(to_replace=-9999, value=np.nan, inplace=True)

    # Do naming conversions
    # If no external names supplied, use defaults
    if not external_names:
        subset_list = list(_define_default_internal_names().values())
        if not 'Fc_storage' in df.columns: subset_list.remove('Fc_storage')
        assert all([x in df.columns for x in subset_list])
        return df[subset_list]

    # If external names, map to internal names and rename
    temp_names = _define_default_internal_names()
    assert all([x in temp_names for x in external_names])
    temp_names.update(external_names)
    if not temp_names['storage_name'] in ds.variables:
        temp_names.pop('storage_name')
    assert all([temp_names[x] in ds.variables for x in temp_names.keys()])
    internal_names = _define_default_internal_names()
    swap_dict = {temp_names[key]: internal_names[key]
                 for key in temp_names.keys()}
    return df[temp_names.values()].rename(swap_dict, axis=1)
#------------------------------------------------------------------------------