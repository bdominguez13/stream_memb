def parametros():
    """Devuelve los parametros a usar en el scritpt:
    tabla: Nombre de la tabla de datos
    st: Nombre de la corriente
    
    do_bg_model: Calcular (yes/no) modelo de fondo
    N_inf, N_sup: Numero min y (max+1) de gaussianas para el xd
    printBIC: Imprimir (yes/no) grafico del BIC
    
    d_inf, d_sup: Limites en distancia de la corriente 
    
    C11, C22, C33: Valores de la diagonal de la matriz de covarianza de la corriente
    
    d_mean, e_dd: Distancia media y su error
    mu1_mean, mu2_mean: Movimientos propios medios en el frame de la corriente
    e_mu1, e_mu2, cov_mu: Valores de la matriz de cov de los movimientos propios en le frame de la correinte
    
    lim_unif: Valores limite de la distrubucion uniforme de los prior de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d y f
    
    nwalkers, ndim, steps: Numero de caminadores, parametros y pasos para la MCMC
    discard: Numero de pasos descartados por el burn-in
    thin: Numero de pasos cada cuanto tomar un valor
    
    q_min, q_max: percentiles minimo y maximo
    
    """
    
    tabla = 'RRLwithprobthin.fit' #Nombre tabla de datos
    st = 'Pal5-PW19' #Nombre de la corriente

    do_bg_model = 'no' #Calcular (yes/no) modelo de fondo
    N_inf, N_sup = 6, 7 #Numero min y (max+1) de gaussianas para el xd
    printBIC = 'no'
    
    d_inf, d_sup = 18, 25 #Limites en distancia de la corriente

    #Matriz de covarianza del stream
    C11, C22, C33 = 0.05**2, 0.05**2, 0.02**2  #mas/yr, mas/yr, kpc
    
    #Priors
    d_mean, e_dd = 23.6, 0.8
    mu1_mean, mu2_mean = 3.78307899, 0.71613004
    e_mu1, e_mu2, rho_mu = 0.022, 0.025, -0.39
    cov_mu = rho_mu*e_mu1*e_mu2 #rho_xy = sigma_xy/(sigma_x*sigma_y)
    # sigma = np.array([[(e_mu1*10)**2, (cov_mu*100)], [(cov_mu*100), (e_mu2*10)**2]])

    lim_unif = [(-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-20, 15), (-20, 15), (-20, 15), (0, 1)]
    # lim_unif = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-20, 15), (-20, 15), (-20, 15), (0.004, 0.012)]

    #MCMC
    nwalkers, ndim, steps = 104, 13, 2**17
    burn_in, thin = 2**10, 2200
    
    #quantiles
    q_min, q_max = 5, 95
    
    return tabla, st, do_bg_model, printBIC, N_inf, N_sup, printBIC, d_inf, d_sup, C11, C22, C33, d_mean, e_dd, mu1_mean, mu2_mean, e_mu1, e_mu2, cov_mu, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_min, q_max


