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
          dataset-specific variable names (see _define_default_external_names()
          function below for key names); if not user-specified, default names
          will be used, but will fail if these names are not used in the data
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
    fig, ax = plt.subplots(1, figsize = (12, 8))
    fig.patch.set_facecolor('white')
    ax.set_ylabel(r'$R_e\/(\mu mol\/m^{-2}\/s^{-1})$', fontsize = 22)
    ax.set_xlabel('$u_{*}\/(m\/s^{-1})$', fontsize = 22)
    ax.tick_params(axis = 'x', labelsize = 14)
    ax.tick_params(axis = 'y', labelsize = 14)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if ustar_threshold: ax.axvline(ustar_threshold, color='black', lw=0.5)
    ax.plot(means_df.ustar, means_df.Fc, marker='o', mfc='None',
            color = 'black', ls=':', label='Turbulent flux')
    if 'Fc_storage' in noct_df.columns:
        ax.plot(means_df.ustar, means_df.Fc_storage, marker='s', mfc='None',
                color = 'black', ls='-.', label='Storage')
        ax.plot(means_df.ustar, means_df.Fc + means_df.Fc_storage,
                marker = '^', mfc = '0.5', color = '0.5', label='Apparent NEE')
        ax.legend(frameon=False)
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_internal_names():

    return {'flux_name': 'Fc',
            'storage_name': 'Fc_storage',
            'insolation_name': 'Fsd',
            'friction_velocity_name': 'ustar'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _make_df(path_to_file, external_names):

    ds = xr.open_dataset(path_to_file)
    ds = ds.sel(latitude=ds.latitude[0], longitude=ds.longitude[0], drop=True)
    temp_names = _define_default_internal_names()
    assert all([x in temp_names for x in external_names])
    temp_names.update(external_names)
    if not temp_names['storage_name'] in ds.variables:
        temp_names.pop('storage_name')
    assert all([temp_names[x] in ds.variables for x in temp_names.keys()])
    ds = ds[list(temp_names.values())]
    internal_names = _define_default_internal_names()
    swap_dict = {temp_names[key]: internal_names[key]
                 for key in temp_names.keys()}
    df = ds.to_dataframe()
    df.index = df.index.drop_duplicates()
    df.replace(to_replace=-9999, value=np.nan, inplace=True)
    ds.close()
    return df.rename(swap_dict, axis=1)
#------------------------------------------------------------------------------