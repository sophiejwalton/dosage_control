import csv
import os
import string

import pandas as pd
import numpy as np

import holoviews as hv
from holoviews import opts
import bokeh.io
import bootcamp_utils.hv_defaults

import panel as pn
pn.extension()



def make_gridplot(df_tidy, names, p1, p2, to_plot = 'concentration', logx = True,
                 yrange = bokeh.models.Range1d(0, 15)):
    plots = []
    for name in names:
        df_small =  df_tidy.loc[df_tidy['species'] == name, :]
        points = hv.Points(
        data=df_small, kdims=[p1, to_plot], vdims=[p2],
  
        )
        if logx:
            
            points.opts(color = p2, cmap = 'Blues', 
                    logx=True, title = name, colorbar = True, size= 7,
                   frame_height=200, frame_width=200 * 3 // 2,)
       
        else: 
            points.opts(color = p2, cmap = 'Blues', 
                   title = name, colorbar = True, size= 7,
                   frame_height=200, frame_width=200 * 3 // 2,)
       # curve = hv.Curve(data = df_small, kdims=[p1, "concentration"],
         #                vdims=[p2]).opts(logx=True,frame_height=200, frame_width=200 * 3 // 2)
       # overlay = hv.Overlay([points, curve])
        p = hv.render(points)
        if name == 'H':
            p.y_range = yrange
        plots.append(p)
    return bokeh.layouts.gridplot(plots, ncols = 2)


def time_plot(df, p1, p2, logx = True):
    points = hv.Points(
        data=df, kdims=[p1, 'time'], vdims=[p2],
  
        )
    if logx:
        points.opts(color = p2, cmap = 'Reds', 
                    logx=True, title ='time to ss', colorbar = True, size= 7,
                   frame_height=200, frame_width=200 * 3 // 2,)
    else: 
        points.opts(color = p2, cmap = 'Reds', 
                   title = 'time to ss', colorbar = True, size= 7,
                   frame_height=200, frame_width=200 * 3 // 2,)
    p = hv.render(points)
    return p
    
    
    
def hv_plot(df_tidy, to_plot = 'all'):
    if to_plot == 'all':

         
        hv_fig = hv.Curve(df_tidy, 
         kdims = ['time', 'concentration'], 
        vdims = ['species']
        ).groupby('species'
        ).overlay(
        ).opts(frame_height=250,frame_width=250 * 3 // 2)
    else:
        df_small = df_tidy.loc[df_tidy['species'].isin(to_plot), :]
        hv_fig = hv.Curve(df_small, 
         kdims = ['time', 'concentration'], 
        vdims = ['species']
        ).groupby('species'
        ).overlay(
        ).opts(frame_height=250,frame_width=250 * 3 // 2)

    # Take out the Bokeh object
    p = hv.render(hv_fig)
    return p


    
def hv_plot_param(df_tidy, species = 'H', param = 'N'):     
    hv_fig = hv.Curve(df_tidy, 
         kdims = ['time', species], 
        vdims = [param],
        
        ).groupby(param
        ).overlay(
        ).opts(frame_height=250,frame_width=250 * 3 // 2)
    hv_fig.opts(opts.Curve(color=hv.Palette('Viridis'),  width=600))


    # Take out the Bokeh object
    p = hv.render(hv_fig)
    return p

def make_smallplot(df_tidy, names, p1, p2, name = 'H', to_plot = 'concentration', logx = True):
    df_small = df_tidy.loc[df_tidy['species'] == name, :]
    points = hv.Points(
        data=df_small, kdims=[p1, to_plot], vdims=[p2])
    if logx:
        points.opts(color = p2, cmap = 'Blues', 
                    logx=True, title = name, colorbar = True, size= 7,
                   frame_height=200, frame_width=200 * 3 // 2,)
    else: 
        points.opts(color = p2, cmap = 'Blues', 
                   title = name, colorbar = True, size= 7,
                   frame_height=200, frame_width=200 * 3 // 2,)
    p = hv.render(points)
    return p 
    
