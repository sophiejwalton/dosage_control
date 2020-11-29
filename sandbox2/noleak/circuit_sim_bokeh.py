import csv
import os
import string
import math
import scipy
import scipy.integrate
import warnings
from autograd.scipy.integrate import odeint
import autograd.numpy as agnp
import numdifftools as nd
import copy
import pandas as pd
import numpy as np
from holoviews import opts
import bootcamp_utils.hv_defaults
import numdifftools as nd
import bokeh.io

import holoviews as hv


import panel as pn
pn.extension()


bokeh.io.output_notebook()
hv.extension('bokeh')

dark_safe_colors = ["#1b9e77", "#d95f02", "#7570b3"]
light_safe_colors = ["#66c2a5", "#fc8d62", "#8da0cb"]

import IPython.display


from collections import namedtuple
SimulationData = namedtuple("SimulationData", ["params", "ts", "solution"])

def moving_max(a, n=3):
    
    sums = np.zeros(len(a) - n)
    st = len(a) - n
    for i in range(st):
    
        sums[i] = np.max(a[i:i + n])
    
        
    return sums

def simulate(system_dv_dt, params, state_inits, t_max, n_times):
    '''
    Run a scipy simulation. 
    
    Params:
        system_dv_dt: A function that takes the parameters xs and params, in that order,
                        where xs is a state vector of the system and params is a list of 
                        parameters, and returns the derivative of the system.
        params: A (numpy array) vector of parameters.
        state_inits: A (numpy array) vector of initial states.
        t_max: Time to simulate out to, in (arbitrary) seconds.
        n_times: The number of points to record. The higher this number, the higher
                    the time resolution of the simulation.
                    
    Returns: A SimulationData object, which is a named tuple containing params,
                ts (times, as a n_times-long vector), and solution (as a <# species>x<n_times>
                numpy array).      
    '''
    t0 = 0
    dt = (t_max - t0) / n_times
    ts = np.linspace(t0, t_max, n_times)

    def dv_dt(t, xs):
        return system_dv_dt(xs, params)
    
    ode = scipy.integrate.ode(dv_dt).set_integrator('lsoda')#, method='bdf', order = 5)
    ode.set_initial_value(state_inits, t0)
    solution = np.zeros((n_times, len(state_inits)))
    solution[0,:] = state_inits
    i = 1
    while ode.successful() and ode.t < t_max and i < n_times:
        solution[i,:] = ode.integrate(ode.t+dt)
        i += 1

    return SimulationData(params = params, ts = ts, solution = solution)

def simple_sim_plot(sim, names, normalize = False, title = ""):
    plt.figure()
    for i in range(sim.solution.shape[1]):

        print(i, names[i])
        if normalize:
    
        
            plt.plot(sim.ts, sim.solution[:,i]/np.max(sim.solution[:,i]), label = names[i])
        else:
            plt.plot(sim.ts, sim.solution[:,i], label = names[i]) 
    plt.xlabel("Time (s, arbitrary)")
    plt.ylabel("Concentration (nM, arbitrary)")
    plt.title(title)
#     plt.ylim([0, 50])
    plt.legend()
    plt.show()    


def pos_hill(X, k, n):
    Xn = X**n
    kn = k**n
    
    return Xn/(Xn + kn)

def neg_hill(X, k, n):
    Xn = X**n
    kn = k**n
    return kn/(Xn + kn)


def pos_hill_or(x, z, kx, kz, n):
    xn = (x/kx)**n
    zn = (z/kz)**n
    return (xn + zn)/(1 + xn + zn)


def pos_hill_and(x, z, kx, kz, n):
    xn = (x/kx)**n
    zn = (z/kz)**n
    return (xn * zn)/(1 + xn * zn)


def find_ss_opt(dv_dt_generator, params, state_inits, debug = False, 
                max_iters = 5000, tol = 1e-3, verbose=False):
        '''
        Calculate the (non-zero) steady-state vector, if one can
        be found, using Scipy to find zeroes of the ODEs for the system.

        params (partial):
            init_state: Initial condition for search. 
            max_iters: If a steady state value is found with negative
                        concentration of any species (or zero concentration of
                        all species), a new seed will be chosen and the
                        algorithm will try again, up to max_iters times.
            verbose:   Determines whether certain warnings will be printed.

        Searches from init_state (within bounds) and looks for a root.
        If it's zero, repeat some number of times before giving up. Raises an
        error if no steady state value found.
        '''
        dv_dt = dv_dt_generator(params)

        def opt_func(state):
            return dv_dt(state)

        def try_optimization(init_state):
            # Define function for calculating Jacobian
            jacobian_at = nd.Jacobian(opt_func)
#             with warnings.catch_warnings():
#                 warnings.filterwarnings("error",
#                     message = "The iteration is not making good progress.*")
#                 try:
            ss_vector, info, ier, mesg = \
                    scipy.optimize.fsolve(opt_func, state_inits,
                                          fprime = jacobian_at, 
                                          full_output = 1)
           # ss = optimize.root(opt_func,  [0, 0], jac=jac, method='hybr')
            #ss_vector = ss.x
            if verbose:
                print("ss_vector: " + str(ss_vector))
                print("info: " + str(info))
                print("ier: " + str(ier))
                print("mesg: " + str(mesg))
            if np.any(np.abs(info['fvec']) > tol):
                if verbose:
                    print("fvec too high: " + str(info['fvec']))
                return None
#                 except RuntimeWarning as e:
#                     if verbose:
#                         print(e)
#                     return None

            # Screen out solutions with negative concentrations, fudging
            # anything that's clearly numerical error to 0.
            if np.any(ss_vector < 0):
                if np.any(ss_vector < -1e-8):
                    if verbose:
                        print("Steady state vector has negative values: " + str(ss_vector))
                    return None
                ss_vector = np.clip(ss_vector, a_min = 0, a_max = None)
            return ss_vector
        
        solution = try_optimization(state_inits)
        if verbose:
            print("solution: " + str(solution))
        if solution is not None:
            if debug:
                plt.clf()
                sim = simulate(dv_dt_generator, params, state_inits, 
                               t_max = 50000, n_times = 10000)
                for i in range(sim.solution.shape[1]):
                    plt.plot(sim.ts, sim.solution[:,i], label = names[i])
                plt.axhline(solution[0], color = "black")
                plt.xlabel("Time (s, arbitrary)")
                plt.ylabel("Concentration (nM, arbitrary)")
                plt.title("Open-loop example trace")
            #     plt.ylim([0, 50])
                plt.legend()
                plt.show()
            return solution
        else:
            raise ValueError("Unable to find non-zero steady state of function!")

def double_param_search_ss3(param2, dvdt, state_inits, default_params, names, title,
                        param1 = 'N',
                        param1s = np.logspace(4, 8, 10), 
                        normalize = False,
                  
                        param2s = np.logspace(-2, 2, 10), tmax = 500, n_times = 1000, log_y = False):

    new_solutions = np.zeros((len(param2s), 
                              len(param1s), len(state_inits)))
    new_solutions_2 = np.zeros((len(param1s)*len(param2s), 2 + len(state_inits)))
   
    k = 0 
    for j, p2 in enumerate(param2s):
        temp_params = copy.copy(default_params)
        temp_params[param2] = p2
        for i, p1 in enumerate(param1s):
            temp_params[param1] = p1
           # simdata = simulate(dv_dt, 
               #        temp_params, state_inits,
                  #     t_max = tmax, n_times = n_times)
             
           # sim = simdata.solution[-int(new_solutions.shape[0]/5):,:]
           # print(p1, p2)
            soln = find_ss_opt(lambda params: (lambda x: dvdt(x, params)), temp_params, state_inits,
                              verbose = False)
            
            new_solutions[j, i,:] = soln # simdata.solution[-int(new_solutions.shape[0]/5):,:].mean(axis = 0) 
            new_solutions_2[k, 0] = p1
            new_solutions_2[k, 1] = p2
            
          
            new_solutions_2[k, 2:] = new_solutions[j, i,:]
            
            k = k + 1 
        
    df = pd.DataFrame(new_solutions_2, columns = [param1, param2] + names)
   
    df_tidy = pd.melt(df, value_vars = names, id_vars = [param1, param2],
                  value_name = 'concentration',
                 var_name = 'species')

    return df_tidy 


def find_ss_sim(dv_dt_generator, params, state_inits, dt = 0.1, debug = False, 
                max_n = 60000, tol = 1e-5):
    t0 = 0
    dv_dt = dv_dt_generator(params)
    
    ode = scipy.integrate.ode(dv_dt).set_integrator('lsoda')#, method='bdf', order = 5)
    ode.set_initial_value(state_inits, t0)
    last_time = np.zeros(len(state_inits))
    i = 0
    broken = False
    while ode.successful():
        last_time = ode.integrate(ode.t+dt)
        i += 1
        if np.all(dv_dt(0, last_time) < tol):
            break
        if i >= max_n:
            broken = True
            break
    if debug:
        plt.clf()
        sim = simulate(dv_dt_generator, params, state_inits, t_max = dt*i, n_times = 10000)
        for i in range(sim.solution.shape[1]):
            plt.plot(sim.ts, sim.solution[:,i], label = names[i])
        plt.axhline(last_time[0], color = "black")
        plt.xlabel("Time (s, arbitrary)")
        plt.ylabel("Concentration (nM, arbitrary)")
        plt.title("Open-loop example trace")
    #     plt.ylim([0, 50])
        plt.legend()
        plt.show()
        
    if broken:
        raise Exception("Max iterations reached without finding steady state.")
    return last_time



def exact_binding_eq(conc_1, conc_2, k, tol = 1e-6):
    '''
    Calculates amount of bound species for two monomerically-binding species, i.e.:
    
    1 + 2 <-->(k) complex
    '''
    a = 1/k
    b = -1 - (conc_1 + conc_2)/k
    c = conc_1 * conc_2 / k
    
    sol1 = (-b + np.sqrt(b**2 - 4*a*c)) / 2 / a
    sol2 = (-b - np.sqrt(b**2 - 4*a*c)) / 2 / a
    
    if isinstance(conc_1, np.ndarray) and isinstance(conc_2, np.ndarray):
        return np.array([exact_binding_eq(conc_1[i], conc_2[i], k, tol) for i in range(len(conc_1))])
    if isinstance(conc_1, np.ndarray):
        return np.array([exact_binding_eq(c1, conc_2, k, tol) for c1 in conc_1])
    if isinstance(conc_2, np.ndarray):
        return np.array([exact_binding_eq(conc_1, c2, k, tol) for c2 in conc_2])
    max_possible = min(conc_1, conc_2)
    if sol1 <= max_possible and sol1 > tol and sol2 <= max_possible and sol2 > tol:
        raise ValueError("Multiple possible solutions with conc_1 = %f, conc_2 = %f, k = %f: %f and %f" \
                         % (conc_1, conc_2, k, sol1, sol2))
    if math.isnan(sol1) or math.isnan(sol2):
        print("Found nan solution with:")
        print("\tconc_1: " + str(conc_1))
        print(f"\tconc_2: {conc_2}")
        print(f"\tk: {k}")
        print(f"\tn: {n}")
        print(f"\tk_f: {k_f}")
        print(f"\tk_r: {k_r}")
        print(f"\ttol: {tol}")
        if n == 2:
            print(f"\tk_term: {k_term}")
        print(f"\ta: {a}")
        print(f"\tb: {b}")
        print(f"\tc: {c}")
        print(f"\tsol1: {sol1}")
        print(f"\tsol2: {sol2}")
    if sol1 < max_possible and sol1 > tol:
        return sol1
    else:
        return sol2






def bound(A, max = 1000):
    return np.max(np.min((A, 1000)), 0)

def get_df(simdata, names, normalize = False):
    simdata_array = np.copy(simdata[2])
    if normalize: 
        for i in range(len(names)):
            simdata_array[:,i ] = simdata_array[:, i] / np.max(simdata_array[:,i])
    df_sim = pd.DataFrame(simdata_array, columns = names)
    df_sim['time'] = simdata[1]

    df_tidy = pd.melt(df_sim, value_vars = names, id_vars = ['time'],
                  value_name = 'concentration',
                 var_name = 'species')
    
        
    return df_tidy
def plot_param_trajectories(system, state_inits, names, params, 
                            species = 'H', 
                            param = 'N',
                           param_range = np.logspace(-2, 1, 5), tmax = 500 ):
    cmap = bokeh.palettes.Viridis[5]
    p = bokeh.plotting.figure(plot_width=400, plot_height=400,
                             title = param
                             )
    p.xaxis.axis_label = 'time'
    p.yaxis.axis_label = 'concentration'
    ind = names.index(species)
    for i, par in enumerate(param_range):
        temp_params = copy.copy(params)
        temp_params[param] = par
        sim = simulate(system, 
                   temp_params, state_inits, 
                   t_max = tmax, n_times =1000)
   
        p.line(x = sim.ts, y = sim.solution[:, ind], color = cmap[i])
    
    color_mapper = bokeh.models.LogColorMapper(palette="Viridis256", low=1e-2, high=1e1)
    
    color_bar = bokeh.models.ColorBar(color_mapper=color_mapper, ticker=bokeh.models.LogTicker(),
                     label_standoff=12, border_line_color=None, location=(0,0))
    p.add_layout(color_bar, 'right')
   
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
    




def double_param_search(param2, dv_dt, state_inits, default_params, names, title,
                        param1 = 'N',
                        param1s = np.logspace(4, 8, 10), 
                        normalize = False,
                  
                        param2s = np.logspace(-2, 2, 10), tmax = 500, n_times = 1000, log_y = False):

    new_solutions = np.zeros((len(param2s), 
                              len(param1s), len(state_inits)))
    new_solutions_2 = np.zeros((len(param1s)*len(param2s), 2 + len(state_inits)))
   
    k = 0 
    for j, p2 in enumerate(param2s):
        temp_params = copy.copy(default_params)
        temp_params[param2] = p2
        for i, p1 in enumerate(param1s):
            temp_params[param1] = p1
            simdata = simulate(dv_dt, 
                       temp_params, state_inits,
                       t_max = tmax, n_times = n_times)
             
            sim = simdata.solution[-int(new_solutions.shape[0]/5):,:]
            
            new_solutions[j, i,:] = simdata.solution[-int(new_solutions.shape[0]/5):,:].mean(axis = 0) 
            new_solutions_2[k, 0] = p1
            new_solutions_2[k, 1] = p2
          
            new_solutions_2[k, 2:] = new_solutions[j, i,:]
            
            k = k + 1 
        
    df = pd.DataFrame(new_solutions_2, columns = [param1, param2] + names)
   
    df_tidy = pd.melt(df, value_vars = names, id_vars = [param1, param2],
                  value_name = 'concentration',
                 var_name = 'species')

    return df_tidy    



def get_param_trajectories(system, state_inits, names, params, 
                            species = 'H', 
                            param = 'N',
                           param_range = np.logspace(-2, 1, 5), tmax = 500 ):

 
    ind = names.index(species)
    dfs = []
    for i, par in enumerate(param_range):
        temp_params = copy.copy(params)
        temp_params[param] = par
        sim = simulate(system, 
                   temp_params, state_inits, 
                   t_max = tmax, n_times =1000)
   
  
        
        df_small = pd.DataFrame(data = {'time': sim.ts, species: sim.solution[:, ind]})
        df_small[param] = par
        dfs.append(df_small)
    
   
   
    return pd.concat(dfs)

    
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



def double_param_search_ss2(param2, dv_dt, state_inits, default_params, names, title,
                        param1 = 'N',
                        param1s = np.logspace(4, 8, 10), 
                        normalize = False,
                  
                        param2s = np.logspace(-2, 2, 10), tmax = 500, n_times = 1000, log_y = False):

    new_solutions = np.zeros((len(param2s), 
                              len(param1s), len(state_inits)))
    new_solutions_2 = np.zeros((len(param1s)*len(param2s), 2 + len(state_inits)))
   
    k = 0 
    for j, p2 in enumerate(param2s):
        temp_params = copy.copy(default_params)
        temp_params[param2] = p2
        for i, p1 in enumerate(param1s):
            temp_params[param1] = p1
           # simdata = simulate(dv_dt, 
               #        temp_params, state_inits,
                  #     t_max = tmax, n_times = n_times)
             
           # sim = simdata.solution[-int(new_solutions.shape[0]/5):,:]
            soln = find_ss_opt(dv_dt_generator, default_params, state_inits)
            
            new_solutions[j, i,:] = soln # simdata.solution[-int(new_solutions.shape[0]/5):,:].mean(axis = 0) 
            new_solutions_2[k, 0] = p1
            new_solutions_2[k, 1] = p2
          
            new_solutions_2[k, 2:] = new_solutions[j, i,:]
            
            k = k + 1 
        
    df = pd.DataFrame(new_solutions_2, columns = [param1, param2] + names)
   
    df_tidy = pd.melt(df, value_vars = names, id_vars = [param1, param2],
                  value_name = 'concentration',
                 var_name = 'species')

    return df_tidy 




def get_plateau(paper_dvdt, default_params, state_inits, names, param_vary = 'gamma',
                solve = False, species = 'H', thresh = .9, 
                param_range = np.logspace(np.log10(.005), np.log10(.02), 10),
                param_N_range = np.logspace(-2, 1, 10) ):
    if solve:
        pass
    else: 
        df_tidy = double_param_search(param_vary, paper_dvdt, state_inits, 
                               default_params, names,
                              
                               param_vary,
                               log_y = False,
                               normalize = False,
                           
                            
                               
                               param1s = param_N_range,
                               param2s = param_range,
                               tmax = 5000, n_times = 10000,
                              )
    df_tidy_small = df_tidy.loc[df_tidy['species'] == species, :]
    plateau_Ns = np.zeros(len(df_tidy[param_vary].unique()))
    unique_Ns = df_tidy['N'].unique()
    for i, p in enumerate(df_tidy[param_vary].unique()):
        df_tidy_p= df_tidy_small.loc[df_tidy_small[param_vary] == p, :]
        concs = df_tidy_p['concentration'].values
        final = df_tidy_p['concentration'].values[-1]
        diffs = np.abs(final - concs)
      #  print(concs)
    
       # print(np.where(concs > thresh*final))
        N_val = unique_Ns[np.where(concs >= thresh*final)[0][0]]
        plateau_Ns[i] = N_val
        
    df_plat = pd.DataFrame(data = {param_vary: df_tidy[param_vary].unique(), 'plat': plateau_Ns})
        
    return df_plat
        
    
def plats_param_search(paper_dvdt, default_params, state_inits, names,
                param, param_vals, 
                solve = False, species = 'H', thresh = .9, 
                param_range = np.logspace(np.log10(.005), np.log10(.02), 10),
                param_N_range = np.logspace(-2, 1, 10)):
    dfs = []
    for p in param_vals:
        temp_params = copy.copy(default_params)
        temp_params[param] = p 
        df_plat = get_plateau(paper_dvdt, temp_params, state_inits, names,
                solve = solve, species = species, thresh = thresh, 
                param_range = param_range,
                param_N_range = param_N_range )
        df_plat[param] = p
        dfs.append(df_plat)
    return pd.concat(dfs)
        
        
        
        
def get_gamma_rob(paper_dvdt, default_params, state_inits, names, param_vary = 'N',
                solve = False, species = 'H', thresh = .9, 
                param_range = np.logspace(np.log10(.005), np.log10(.02), 10),
                param_N_range = np.logspace(-2, 1, 10) ):
    if solve:
        pass
    else: 
        df_tidy = double_param_search('gamma', paper_dvdt, state_inits, 
                               default_params, names,
                                     
                              
                               'gamma',
                               param1 = param_vary,
                               log_y = False,
                               normalize = False,
                           
                            
                               
                               param1s = param_N_range,
                               param2s = param_range,
                               tmax = 5000, n_times = 10000,
                              )
    df_tidy_small = df_tidy.loc[df_tidy['species'] == species, :]
    rob_gammas = np.zeros(len(df_tidy[param_vary].unique()))
   
    for i, p in enumerate(df_tidy[param_vary].unique()):
        df_tidy_p= df_tidy_small.loc[df_tidy_small[param_vary] == p, :]
        concs = df_tidy_p['concentration'].values
        # should I change this?? 
        diff = np.max(concs)/np.min(concs) # - np.min(concs))/(np.mean(concs)) 
   
        rob_gammas[i] = diff
        
    df_plat = pd.DataFrame(data = {param_vary: df_tidy[param_vary].unique(), 'rob_gamma': rob_gammas})
        
    return df_plat


def gamma_stuff(param1, param2, paper_dvdt, default_params, state_inits, names, 
                    param_vary = 'N', 
                          param_range = np.logspace(np.log10(.005), np.log10(.02), 10),
                param_N_range = np.logspace(-1, 2, 15), ar_range = np.logspace(-1, 2, 15),
                include_original_params = False,
               ):
    dfs = []
    


    temp_params = copy.copy(default_params)
    if include_original_params:
        param_vary = param2
        plats = get_gamma_rob(paper_dvdt, temp_params, state_inits, names, thresh = .95, 
                    param_vary = param_vary, 
                          param_range = param_range,
                param_N_range =  param_N_range
                          
                          
                    )
        plats[param1] = temp_params[param1]
        dfs.append(plats)
    
    
        param_vary = param1
        plats = get_gamma_rob(paper_dvdt, temp_params, state_inits, names, thresh = .95, 
                    param_vary = param_vary, 
                          param_range = param_range,
                param_N_range =  param_N_range
                          
                          
                    )
        plats[param2] = temp_params[param2]
        dfs.append(plats)
    
    
    param_vary = param1
    for v in ar_range: 
        
        temp_params[param2] = v
   # print(temp_params)
        
        plats = get_gamma_rob(paper_dvdt, temp_params, state_inits, names, thresh = .95, 
                    param_vary = param_vary, 
                          param_range = param_range,
                param_N_range =  param_N_range
                          
                          
                    )
        plats[param2] = v
        dfs.append(plats)
    
    df_full = pd.concat(dfs)

    #df_full.to_csv(f'{circuit_name}_df_gamma_rob_{param1}_{param2}_layered.csv')
    return df_full




        
def get_plat_rob(paper_dvdt, default_params, state_inits, names, param1, param2,
                solve = False, species = 'H', thresh = .95, 
                param1_range = np.logspace(-2, 2, 10),
                param2_range = np.logspace(-2, 2, 10),
                param_N_range =np.logspace(-1, 2, 20)
                ):
    if solve:
        pass
    else: 
        dfs = []
        vmax_range = np.logspace(-2, 2, 10)
        temp_params = copy.copy(default_params)
        for v in vmax_range: 
            temp_params[param1] = v # param1
   # print(temp_params)
            param_vary = param2
            plats = get_plateau(paper_dvdt, temp_params, state_inits, names, thresh = .95, 
                    param_vary = param_vary,
                    
                   #param_N_range = np.linspace(.05, 1, 20),
                   param_N_range = np.logspace(-1, 2, 20),
                   param_range = np.logspace(-2, 2, 10))
            plats[param1] = v
            dfs.append(plats)
    
        df_full = pd.concat(dfs)   
    return df_full




def get_overshoot(simdata, names, species = 'H', eps_ratio = .02, span = 10, normalize = True ):
    ind = names.index(species)

    data = simdata.solution[:, ind ]
    

    end_point = np.mean(data[-span:])
    #print(names[ind], simdata.solution.shape, end_point, np.max(data))
    max_value = np.max(data)

    return max_value/end_point

def stimulus_species(simdata, system, params, names, 
                 tmax, ntimes = 1000, to_botch = ['H'], factor = .5):
    end_point = np.copy(simdata.solution[-1, :])
   
    for name in to_botch: 
        ind = names.index(name)
        end_point[ind] = end_point[ind] + end_point[ind]*factor
 
    simdata_2 = simulate(system, 
                    params, end_point, 
                   t_max = tmax, n_times = ntimes)
    return simdata_2


    
def double_param_overshoot(param2, dv_dt, state_inits, default_params, names, title,
                        param1 = 'N',
                        param1s = np.logspace(4, 8, 10), 
                  
                           eps_ratio = .02, 
                           normalize = True, 
                           species = 'H',
                  
                        param2s = np.logspace(-2, 2, 10), tmax = 500, n_times = 1000, log_y = False):


    new_solutions_2 = np.zeros((len(param1s)*len(param2s), 3))
    
    k = 0 
    for j, p2 in enumerate(param2s):
        temp_params = copy.copy(default_params)
        temp_params[param2] = p2
        for i, p1 in enumerate(param1s):
            temp_params[param1] = p1
            simdata = simulate(dv_dt, 
                       temp_params, state_inits,
                       t_max = tmax, n_times = n_times)
            
           # simdata_recovery = stimulus_species(simdata, dv_dt, temp_params, names, 
                 #tmax, ntimes = 1000, to_botch = ['H'], factor = .5)
            
            over_shoot = get_overshoot(simdata, names, species = species, eps_ratio = eps_ratio,  normalize = normalize )
             
            new_solutions_2[k, 0] = p1
            new_solutions_2[k, 1] = p2
          
            new_solutions_2[k, 2] = over_shoot
            
            k = k + 1 
        
    df = pd.DataFrame(new_solutions_2, columns = [param1, param2, 'over'])

    return df

def plot_sensitivities(df_tidy, name = 'H', logx = True, plot_abs = False, param = 'N'):
    df_tidy_n =  df_tidy.loc[df_tidy['species'] == name, :]
    plots = []
    for sense in [param, 'gamma']:
    
        df_small = df_tidy_n.loc[df_tidy_n['sensitivity_to'] == sense, :]
        if plot_abs:
            df_small['abs_sensitivity'] = np.abs(df_small['sensitivity'].values)

            points = hv.Points(
            data=df_small, kdims=[param, 'abs_sensitivity'], vdims=['gamma'],
  
            )
        else: 
            points = hv.Points(
            data=df_small, kdims=[param, 'sensitivity'], vdims=['gamma'],
  
            )


        points.opts(color = 'gamma', cmap = 'Purples', logx = logx,
                   title = sense, colorbar = True, size= 7,
                   frame_height=200, frame_width=200 * 3 // 2,)
        p = hv.render(points)
        plots.append(p)
    
    return plots

def double_sensitivity_params(paper2_dvdt, default_params, names,param = 'N',
                              param_N_range = np.logspace(-2, 1, 10), 
                              param_g_range = np.logspace(np.log10(.005), np.log10(.02), 10),
                              
                              tspan2 = agnp.linspace(0, 2000.), 
                             x0 = (0,0,0)):

    
    def C2(K):
        gamma, N = K 
     


        def dC2dt(xs, t, gamma, N):
        
            return paper2_dvdt(xs, gamma, N, default_params)
        sol = odeint(dC2dt, x0, tspan2, tuple((gamma, N)))
        return sol
    
    
    k = 0 
    gamma_sensitivities = np.zeros((len(param_N_range)*len(param_g_range), 2 + len(names)))
    N_sensitivities = np.zeros((len(param_N_range)*len(param_g_range), 2 + len(names)))

    for j, p2 in enumerate(param_g_range):

        for i, p1 in enumerate(param_N_range):
            
            K2 = [p2, p1]
           # print(K2)

             
            sensitivities = nd.Jacobian(C2)(K2).T
            
            gamma_sensitivities[k, 2:] = sensitivities[:, 0, -1]
            N_sensitivities[k, 2:] = sensitivities[:, 1, -1]
            gamma_sensitivities[k, 0] = p1
            N_sensitivities[k, 0] = p1                               
            gamma_sensitivities[k, 1] = p2
            N_sensitivities[k, 1] = p2                                 

            k = k + 1 
        
    df_g = pd.DataFrame(gamma_sensitivities, columns = [param, 'gamma'] + names)
    df_g['sensitivity_to'] = 'gamma'
    df_N = pd.DataFrame(N_sensitivities, columns = [param, 'gamma'] + names)
    df_N['sensitivity_to'] = 'N'
                               
    df_tidy_g = pd.melt(df_g, value_vars = names, id_vars = [param, 'gamma'],
                  value_name = 'sensitivity',
                 var_name = 'species')
    df_tidy_g['sensitivity_to'] = 'gamma'
    df_tidy_N = pd.melt(df_N, value_vars = names, id_vars = [param, 'gamma'],
                  value_name = 'sensitivity',
                 var_name = 'species')
    df_tidy_N['sensitivity_to'] = 'N'
    return pd.concat([df_tidy_N, df_tidy_g])   


def time_to_ss(simdata, names, species = 'H', eps_ratio = .02, span = 50):
    ind = names.index(species)
    #print(species, names, ind)
    data = simdata.solution[:, ind]
    end_point = np.mean(data[-span:])
    
    changes = np.abs(data - end_point)
   # print(changes)
    
    
    error_moving_max = moving_max(changes, n = span)
    #print(error_moving_avg.shape)
    ss_ind = np.where(error_moving_max < eps_ratio*end_point)[0][0]
    ss_time = simdata.ts[ss_ind]
   # print('k', ss_time, ss_ind, np.max(data[ss_ind:ss_ind + 50]), np.min(data[ss_ind:ss_ind + 50]), end_point)
  #  print(end_point, data[ss_ind])
    return ss_time, ss_ind, end_point

def double_param_search_times(param2, dv_dt, state_inits, default_params, names, title,
                        param1 = 'N',
                        species = 'H',
                        param1s = np.logspace(4, 8, 2), 
                        normalize = False,
                  
                        param2s = np.logspace(-2, 2, 2),
                              tmax = 1000, n_times = 1000, stimulate = False, 
                              log_y = False):

    new_solutions = np.zeros((len(param2s), 
                              len(param1s), 2))
    new_solutions_2 = np.zeros((len(param1s)*len(param2s), 2 + 2))
   
    k = 0 
    for j, p2 in enumerate(param2s):
        temp_params = copy.copy(default_params)
        temp_params[param2] = p2
        for i, p1 in enumerate(param1s):
            temp_params[param1] = p1
            
            simdata = simulate(dv_dt, 
                       temp_params, state_inits,
                       t_max = tmax, n_times = n_times)
            
            if stimulate: 
                simdata = stimulus_species(simdata, dv_dt, temp_params, names, 500)
            sim = simdata.solution[-int(new_solutions.shape[0]/5):,:]
            
            
            
            ss_time, ss_ind, end_point = time_to_ss(simdata,
                                                    names, species = 'H',
                                                    span = 50
                                               )
          #  if ss_time < 50:
               # print(p1, p2, ss_time, end_point, sim[-1])
           
            
            new_solutions_2[k, 0] = p1
            new_solutions_2[k, 1] = p2
            new_solutions_2[k, 2] = ss_time
           # print(p1, p1)
          
            new_solutions_2[k, 3] = end_point
            
            k = k + 1 
        
    df = pd.DataFrame(new_solutions_2, columns = [param1, param2, 'time', species])
   
   # df_tidy = pd.melt(df, value_vars = names, id_vars = [param1, param2],
          #        value_name = 'concentration',
          #       var_name = 'species')

    return df

