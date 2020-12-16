def double_sensitivity_params_old(paper2_dvdt, default_params, names,param = 'N',
                              param_N_range = np.logspace(-2, 1, 10), 
                              param_g_range = np.logspace(np.log10(.005), np.log10(.02), 10),
                              
                              tspan2 = agnp.linspace(0, 4000., 1000), 
                             x0 = (0,0,0)):
    
    '''
    
    Finds the sensitivity of species for a system of ODEs paper2_dvdt to N and gamma. 
    
    Params:

        paper2_dvdt: A function that takes the parameters xs, gamma, N, d params (which do not 
        include gamma and N), in that order,
                        where xs is a state vector of the system and params is a list of 
                        parameters, and returns the derivative of the system.
        param_N_range : parameters to simulate for population size
        param_g_range: parameters to simulate for dilution rates 
        x0: initial state
        default_params: A dictionary of parameters
        names: Names of species in the state vector

        
    Returns: A tidy data frame of sensitivities of each species to gamma and N 
    '''


    
    def C2(K):
        gamma, N = K 
     


        def dC2dt(xs, t, gamma, N):
        
            return paper2_dvdt(xs, gamma, N, default_params)
        sol = odeint(dC2dt, x0, tspan2, tuple((gamma, N)))
      #  sol = np.mean(full_sol[]
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
          #  print(np.abs(sensitivities[1, 0, -100:10]))
          #  print(p2,p1, np.var(sensitivities[1, 0, -10:]), np.var(sensitivities[1, 0, -10:]) > 1)
           # if np.var(sensitivities[1, 0, -10:]) > 1:
              #  print('bad')
            
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
