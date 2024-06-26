import glob 
import os
from astropy.io import ascii, fits
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.io import readsav
from os.path import dirname, join as pjoin
import scipy.io as sio
import matplotlib.colors as colors
import matplotlib as mpl
import pygraphviz as pgv
from networkx.drawing.nx_agraph import graphviz_layout
import numpy.polynomial.polynomial as poly
# from spectres import spectres, spectral_resampling

from astropy.modeling.models import BlackBody
import astropy.units as u
from scipy.optimize import curve_fit

import matplotlib.gridspec as gridspec
from matplotlib.ticker import AutoMinorLocator


def add_to_dict(sav_data, cat, obid):
    '''
    This function strips the content in a sav file and
    puts it in a dictionary
    Input:
        - sav_data: the Save_file from readsav()
        - cat: dictionary
        - obid: an object name of your choice
    Returns: 
        - cat: dictionary with a new entry added
    '''
    def _add_to_dict(sav_data,obid, cat, ourkey):
        shortkey=ourkey
        if 'final_' in ourkey:
            shortkey = ourkey.split('final_')[1]
        try:
            data = sav_data[ourkey]
            cat[obid][shortkey] = data
        except Exception as e:
            cat[obid][shortkey] = None
            print(obid ,":", repr(e))
        return cat

    cat = _add_to_dict(sav_data, obid,  cat, 'final_wave')
    cat = _add_to_dict(sav_data, obid,  cat, 'final_spec')
    cat = _add_to_dict(sav_data, obid,  cat, 'final_specerr')
    cat = _add_to_dict(sav_data, obid,  cat, 'final_phot_wave')
    cat = _add_to_dict(sav_data, obid,  cat, 'final_phot_fnu')
    cat = _add_to_dict(sav_data, obid,  cat, 'final_spec_irs')
    cat = _add_to_dict(sav_data, obid,  cat, 'scale_photosphere_whole')
    cat = _add_to_dict(sav_data, obid,  cat, 'mips_24_factor_scaling')
    cat = _add_to_dict(sav_data, obid,  cat, 'mips_24')
    cat = _add_to_dict(sav_data, obid,  cat, 'ubv_cat')

#         add_to_dict(sav_data, obid, 'final_bbody')
#         add_to_dict(sav_data, obid, 'final_contfit_full')

    return cat

def fit_cont(spectrum, wave_grid=None, flux_downSamp_switch=True, regions=None, order=None):
    """
    Fit continuum to a spectrum.

    Args:
        spectrum (dict): Dictionary containing the spectrum data with keys 'wave' and 'IRS_cont_div'.
        wave_grid (array-like, optional): Grid of wavelengths for interpolation. If not provided, use the original wave array from the spectrum. Default is None.
        flux_downSamp_switch (bool, optional): Switch for flux downsampling. If True, interpolate flux onto wave_grid. If False, use original flux array. Default is True.
        regions (list, optional): List of tuples specifying the wavelength ranges for the continuum regions. Each tuple should contain two values representing the start and end wavelengths of the region. Default is None, in which case the default regions [(5.61, 7.94), (13.02, 13.50), (14.32, 14.83), (30.16, 32.19), (35.07, 35.92)] will be used.

    Returns:
        tuple: Tuple containing three arrays: wave_grid, cont (continuum), and flux_interp (interpolated flux).
    """
    wave = spectrum['wave']
    flux = spectrum['IRS_cont_div']
    
    if flux_downSamp_switch:
        flux_interp = np.interp(wave_grid, wave, flux)
    else:
        flux_interp = flux
        wave_grid = wave
        
    if regions is None:
        regions = [(5.61, 7.94), (13.02, 13.50), (14.32, 14.83), (30.16, 32.19), (35.07, 35.92)]
    else:
        # Check if smallest and largest values in regions are within range of wave array
        wave_min = np.min(wave)
        wave_max = np.max(wave)
        for region in regions:
            if region[0] < wave_min:
                print("Error: Smallest value in the regions is smaller than the minimum of the wave array.")
                print("Using Smallest value in wave array ...")
                region = list(region)
                region[0] = wave_min
                region = tuple(region)
#                 return None
            if region[1] > wave_max:
                print("Error: Largest value in the regions is larger than the maximum of the wave array.")
                return None

    # Create boolean masks for indexing using an iterative approach
    mask = np.zeros_like(wave_grid, dtype=bool)
    for region in regions:
        mask |= (wave_grid >= region[0]) & (wave_grid <= region[1])

    x = wave_grid[mask]
    y = flux_interp[mask]
    
    anchor_x = x 
    anchor_y = y
    
    coefs = poly.polyfit(x, y, order)
    cont = poly.polyval(wave_grid, coefs)  
    
    diff_per = (flux_interp - cont)/flux_interp*100
    
#     print(np.max(diff_per), np.min(diff_per))
    
    return wave_grid, cont, flux_interp, diff_per, anchor_x, anchor_y, mask


def create_3_panel_plot(x_top_list, y_top_list, x_middle_list, y_middle_list, x_bottom_list, y_bottom_list,
                         custom_labels_top, custom_labels_middle, custom_labels_bottom,
                         x_label_top, y_label_top, x_label_middle, y_label_middle,
                         x_label_bottom, y_label_bottom, fontsize_dict,
                         top_plot_method=None, middle_plot_method=None, bottom_plot_method=None,
                         save_path=None, negative_data_switch=1, xmin=4.5, xmax=37):
    """
    Creates a 3-panel plot with a top, middle, and bottom panel using Matplotlib.

    Args:
        x_top_list (list of array-like): List of x-axis data for the top panel.
        y_top_list (list of array-like): List of y-axis data for the top panel.
        x_middle_list (list of array-like): List of x-axis data for the middle panel.
        y_middle_list (list of array-like): List of y-axis data for the middle panel.
        x_bottom_list (list of array-like): List of x-axis data for the bottom panel.
        y_bottom_list (list of array-like): List of y-axis data for the bottom panel.
        custom_labels_top (list of str): List of custom labels for the top panel.
        custom_labels_middle (list of str): List of custom labels for the middle panel.
        custom_labels_bottom (list of str): List of custom labels for the bottom panel.
        x_label_top (str): X-axis label for the top panel.
        y_label_top (str): Y-axis label for the top panel.
        x_label_middle (str): X-axis label for the middle panel.
        y_label_middle (str): Y-axis label for the middle panel.
        x_label_bottom (str): X-axis label for the bottom panel.
        y_label_bottom (str): Y-axis label for the bottom panel.
        fontsize_dict (dict): Dictionary containing fontsize customization for legend, x-axis labels, and y-axis labels.
                             Example: {'legend': 12, 'x_label': 12, 'y_label': 12}
        top_plot_method (list of str, optional): List of plot methods for the top panel (e.g., 'scatter', 'line').
        middle_plot_method (list of str, optional): List of plot methods for the middle panel.
        bottom_plot_method (list of str, optional): List of plot methods for the bottom panel.
        save_path (str, optional): File path and name for saving the plot as a PDF. Defaults to None (i.e., not saving).
        negative_data_switch (int, optional): Switch for handling negative data in the bottom panel. Defaults to 1.
        xmin (float, optional): Minimum value for the x-axis. Defaults to 4.5.
        xmax (float, optional): Maximum value for the x-axis. Defaults to 37.

    Returns:
        None
    """
    # Set some marker shapes
    marker_top_list = ['-', '-', '-.', '--', 'o']
    marker_middle_list = ['-', 'o']
    marker_bottom_list = ['-']
    # Get the number of panels
    num_top = len(x_top_list)
    num_middle = len(x_middle_list)
    num_bottom = len(x_bottom_list)

    # Create a 3-panel plot with gridspec
    fig = plt.figure(figsize=(12, 18))  # Set figure size
    gs = gridspec.GridSpec(3, 1, height_ratios=[2, 1, 1])  # Set gridspec with 3 rows and 1 column, and height ratio

    # Create top panel
    ax_top = plt.subplot(gs[0])  # Create subplot for top panel
    ax_top.spines[['top', 'right', 'left', 'bottom']].set_visible(True)  # Show borders on all 4 sides
    ax_top.tick_params(which='both', direction='out', top=False, right=False, 
                       labelsize=fontsize_dict['tick_labels'])  # Show major and minor axis ticks and customize tick label fontsize

    ax_top.grid(False)  # Hide gridlines
    ax_top.set_xlabel(x_label_top, fontsize=fontsize_dict['x_label'])  # Set x-axis label with fontsize
    ax_top.set_ylabel(y_label_top, fontsize=fontsize_dict['y_label'])  # Set y-axis label with fontsize
    ax_top.xaxis.set_minor_locator(AutoMinorLocator())  # Enable minor ticks on x-axis
    ax_top.yaxis.set_minor_locator(AutoMinorLocator())  # Enable minor ticks on y-axis
    for i in range(len(x_top_list)):
        if top_plot_method and top_plot_method[i] == 'scatter':
            ax_top.scatter(x_top_list[i], y_top_list[i], marker=marker_top_list[i],
                           label=f'{custom_labels_top[i]}', edgecolors='black', lw=1, zorder=10)  # Use scatter plot if specified
        else:
            ax_top.plot(x_top_list[i], y_top_list[i], marker_top_list[i], label=f'{custom_labels_top[i]}')  # Use plot by default


    ax_top.legend(fontsize=fontsize_dict['legend'])  # Set legend fontsize
    ax_top.set_xlim([xmin, xmax])

    # Create middle panel
    ax_middle = plt.subplot(gs[1], sharex=ax_top)  # Share x-axis with the top panel
    ax_middle.spines[['top', 'right', 'left', 'bottom']].set_visible(True)  # Show borders on all 4 sides
    ax_middle.tick_params(which='both', direction='out', top=False, right=False,
                         labelsize=fontsize_dict['tick_labels']) # Show major and minor axis ticks
    ax_middle.grid(False) # Hide gridlines
    ax_middle.set_xlabel(x_label_middle, fontsize=fontsize_dict['x_label']) # Set x-axis label with fontsize
    ax_middle.set_ylabel(y_label_middle, fontsize=fontsize_dict['y_label']) # Set y-axis label with fontsize
    
    for i in range(len(x_middle_list)):
        if middle_plot_method and middle_plot_method[i] == 'scatter':
            ax_middle.scatter(x_middle_list[i], y_middle_list[i], label=f'{custom_labels_middle[i]}', 
                              edgecolors='black', lw=1, zorder=20)  # Use scatter plot if specified
        else:
            if negative_data_switch == 1:
                mask = y_middle_list[i] < 0  # create mask for negative values
                ax_middle.plot(x_middle_list[i], y_middle_list[i], 
                               marker_middle_list[i],label=f'{custom_labels_middle[i]}', 
                               color='C0', zorder = 2)  # plot remaining data
                ax_middle.scatter(x_middle_list[i][mask], y_middle_list[i][mask], marker = "X", 
                                  label='negative data', color='red', zorder = 3)  # plot negative data
            else:
                ax_middle.plot(x_middle_list[i], y_middle_list[i], marker_middle_list[i] ,
                               label=f'{custom_labels_middle[i]}')  # Use plot by default

    ax_middle.legend(fontsize=fontsize_dict['legend'])  # Set legend fontsize
    ax_middle.xaxis.set_minor_locator(AutoMinorLocator())  # Enable minor ticks on x-axis
    ax_middle.yaxis.set_minor_locator(AutoMinorLocator())  # Enable minor ticks on y-axis
    ax_middle.set_xlim([xmin, xmax])

    # Create bottom panel
    ax_bottom = plt.subplot(gs[2], sharex=ax_top)  # Share x-axis with the top panel
    ax_bottom.spines[['top', 'right', 'left', 'bottom']].set_visible(True)  # Show borders on all 4 sides
    ax_bottom.tick_params(which='both', direction='out', top=False, right=False,
                         labelsize=fontsize_dict['tick_labels']) # Show major and minor axis ticks
    ax_bottom.grid(False) # Hide gridlines
    ax_bottom.set_xlabel(x_label_bottom, fontsize=fontsize_dict['x_label']) # Set x-axis label with fontsize
    ax_bottom.set_ylabel(y_label_bottom, fontsize=fontsize_dict['y_label']) # Set y-axis label with fontsize
    
    for i in range(len(x_bottom_list)):
        if bottom_plot_method and bottom_plot_method[i] == 'scatter':
            ax_bottom.scatter(x_bottom_list[i], y_bottom_list[i], label=f'{custom_labels_bottom[i]}', 
                             zorder = 1)  # Use scatter plot if specified
        else:
            if negative_data_switch == 1:
                mask = y_bottom_list[i] < 0  # create mask for negative values
                ax_bottom.plot(x_bottom_list[i], y_bottom_list[i], 
                               marker_bottom_list[i],label=f'{custom_labels_bottom[i]}', 
                               color='C0', zorder = 2)  # plot remaining data
#                 ax_bottom.scatter(x_bottom_list[i][mask], y_bottom_list[i][mask], marker = "X", 
#                                   label='negative data', color='red', zorder = 3)  # plot negative data
            else:
                ax_bottom.plot(x_bottom_list[i], y_bottom_list[i], marker_bottom_list[i] ,
                               label=f'{custom_labels_bottom[i]}')  # Use plot by default

    ax_bottom.legend(fontsize=fontsize_dict['legend'])  # Set legend fontsize
    ax_bottom.xaxis.set_minor_locator(AutoMinorLocator())  # Enable minor ticks on x-axis
    ax_bottom.yaxis.set_minor_locator(AutoMinorLocator())  # Enable minor ticks on y-axis
    ax_bottom.set_xlim([xmin, xmax])

    
    plt.tight_layout()  # Add tight layout

    if save_path:
        plt.savefig(save_path, format='pdf')  # Save plot as PDF if save_path is provided

    plt.show()  # Show the plot
    return
