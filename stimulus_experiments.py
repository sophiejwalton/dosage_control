
import numpy as np
import scipy.special
import pandas as pd
import scipy
import scipy.integrate
from autograd.scipy.integrate import odeint
import numdifftools as nd
import IPython.display
import autograd.numpy as agnp
from circuit_sim_bokeh import *
from collections import namedtuple
SimulationData = namedtuple("SimulationData", ["params", "ts", "solution"])



def stimulus_N(simdata, system, params, tmax, ntimes = 1000):
    end_point = simdata.solution[-1, :]
    N = params['N']
    temp_params = copy.copy(params)
    temp_params['N'] = N*1.5 
    
    
    
    simdata_2 = simulate(system, 
                    temp_params, end_point, 
                   t_max = tmax, n_times = ntimes)
    
    
    
    return simdata_2



def stimulus_AHL(simdata, system, params, names, 
                 tmax, ntimes = 1000, to_botch = ['H', 'H2']):
    end_point = simdata.solution[-1, :]
    for name in to_botch: 
        ind = names.index(name)
        end_point[ind] = end_point[ind]*2
    simdata_2 = simulate(system, 
                    params, end_point, 
                   t_max = tmax, n_times = ntimes)
    return simdata_2

def stimulus_gamma(simdata, system, params, tmax, ntimes = 1000):
    end_point = simdata.solution[-1, :]
    print(end_point)
    
    gamma = params['gamma']
    temp_params = copy.copy(params)
    temp_params['gamma'] = gamma*2 
    
    
    
    simdata_2 = simulate(system, 
                    temp_params, end_point, 
                   t_max = tmax, n_times = ntimes)
    
    
    
    return simdata_2



def moving_average(a, n=3):
    
    sums = np.zeros(len(a) - n)
    st = len(a) - n
    for i in range(st):
    
        sums[i] = np.max(a[i:i + n])
    
        
    return sums

def time_to_ss(simdata, names, species = 'H', eps_ratio = .02, span = 50):
    ind = names.index(species)
    #print(species, names, ind)
    data = simdata.solution[:, ind]
    end_point = np.mean(data[-span:])
    
    changes = np.abs(data - end_point)
   # print(changes)
    
    
    error_moving_avg = moving_average(changes, n = span)
    #print(error_moving_avg.shape)
    ss_ind = np.where(error_moving_avg < eps_ratio*end_point)[0][0]
    ss_time = simdata.ts[ss_ind]
   # print('k', ss_time, ss_ind, np.max(data[ss_ind:ss_ind + 50]), np.min(data[ss_ind:ss_ind + 50]), end_point)
  #  print(end_point, data[ss_ind])
    return ss_time, ss_ind, end_point



def get_overshoot(simdata, names, species = 'H', eps_ratio = .02, span = 10, normalize = True ):
    ind = names.index(species)
    data = simdata.solution[:, ind]
    end_point = data[-1]
    max_value = np.max(data)
    diff = max_value - end_point
    if normalize: 
        return diff/end_point
    return diff



    
def stimulus_param(simdata, system, params, tmax, 
                   ntimes = 1000, factor = 1.5, param = 'N'):
    end_point = simdata.solution[-1, :]
    p = params[param]
    temp_params = copy.copy(params)
    temp_params[param] = p*factor
    
    
    
    simdata_2 = simulate(system, 
                    temp_params, end_point, 
                   t_max = tmax, n_times = ntimes)
    
    
    
    return simdata_2

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

def plot_compiled_data(simdata_before, simdata_after,
                       names, to_plot = 'H'):
    df_tidy_before = get_df(simdata_before, names, normalize = False)
    
    df_tidy_after = get_df(simdata_after, names, normalize = False)
    
    ts_stim = simdata_before.ts[-1]
    df_tidy_after['time'] = df_tidy_after['time'] + ts_stim
    df_tidy_full = pd.concat([df_tidy_before, df_tidy_after])
    p = hv_plot(df_tidy_full, to_plot = [to_plot])
    p.ray(x = ts_stim, color ='red', angle = 90, angle_units = 'deg')
    end_point = df_tidy_before.loc[df_tidy_before['species'] == to_plot, 'concentration'].values[-1]
    p.ray(y = end_point, color = 'orange')

    return p, df_tidy_full, end_point


def multiple_param_stimulus(simdata, system, params, tmax, names, 
                   ntimes = 1000, factor = 1.5, param = 'N',
                              st_range = np.linspace(.5, 2, 10)):
    times = np.zeros(len(st_range))
    ends = np.zeros(len(st_range))
    changes = np.zeros(len(st_range))
    _, _, start = time_to_ss(simdata, names, 
                            species = 'H',  span = 50 )
    
    for i, st in enumerate(st_range):
        
        stimdata_recovery = stimulus_param(simdata, system, params, tmax, 
                   ntimes = ntimes, factor = st, param = param)
        ss_time, _, end_point = time_to_ss(stimdata_recovery, names, 
                            species = 'H',  span =50 )
        times[i] = ss_time
        ends[i] = end_point 
        changes[i] = np.abs(start - end_point)/start
    df_stim = pd.DataFrame(data = {'param': param,'factor': st_range, 'times': times,
                                   'ends': ends, 'changes': changes})
    return df_stim



def multiple_species_stimulus(simdata, system, params, tmax, names, 
                   ntimes = 1000, factor = 1.5, species = 'H',
                              st_range = np.linspace(-.5, .5, 10)):
    _, _, start = time_to_ss(simdata, names, 
                            species = 'H',  span = 50 )
    times = np.zeros(len(st_range))
    ends = np.zeros(len(st_range))
    changes = np.zeros(len(st_range))
    for i, st in enumerate(st_range):
        
        stimdata_recovery = stimulus_species(simdata, system, params, names, 
                 tmax, ntimes = 1000, to_botch = [species], factor = st)
        ss_time, _, end_point = time_to_ss(stimdata_recovery, names, 
                            species = 'H',  span = 50 )
        
        times[i] = ss_time
        ends[i] = end_point
        changes[i] = np.abs(start - end_point)/start
    df_stim = pd.DataFrame(data = {'species': species, 
                                   'factor': st_range, 'times': times, 'ends': ends, 'changes': changes})
    return df_stim
