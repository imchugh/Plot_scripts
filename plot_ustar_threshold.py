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
    ds = xr.open_dataset(path_to_file)
    df = ds.to_dataframe()
    df.index = df.index.drop_duplicates()
    df.replace(to_replace=-9999, value=np.nan, inplace=True)
    ds.close()

    # Dump extraneous variables and swap keys to internal names
    if not vars_dict: vars_dict = _define_default_internal_names()
    df = df[vars_dict.values()]
    if vars_dict: _rename_df(df, vars_dict, _define_default_internal_names())

    # Group by quantile and generate mean
    noct_df = df.loc[df[vars_dict['insolation_name']] < light_threshold]
    noct_df['ustar_cat'] = pd.qcut(df.ustar, num_cats,
                                   labels = np.linspace(1, num_cats, num_cats))
    means_df = noct_df.groupby('ustar_cat').mean()

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
    if ustar_threshold: ax.axvline(ustar_threshold, color = 'grey')
    ax.plot(means_df[vars_dict['friction_velocity_name']],
            means_df[vars_dict['flux_name']],
            marker = 'o', mfc = '0.5', color = 'grey')
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _define_default_internal_names():

    return {'flux_name': 'Fc',
            'insolation_name': 'Fsd',
            'friction_velocity_name': 'ustar'}
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
def _rename_df(df, external_names, internal_names):
    assert sorted(external_names.keys()) == sorted(internal_names.keys())
    swap_dict = {external_names[key]: internal_names[key]
                 for key in internal_names.keys()}
    sub_df = df[swap_dict.keys()].copy()
    sub_df.columns = swap_dict.values()
    return sub_df
#------------------------------------------------------------------------------