def parametros():
    """Devuelve los parametros a usar en el scritpt:
    tabla: Nombre de la tabla de datos
    st: Nombre de la corriente
    
    do_xd_model: Calcular (yes/no) modelo de fondo
    N_inf, N_sup: Numero min y (max+1) de gaussianas para el xd
    i_best_xd: Mejor numero de gaussianas para el modelo del fondo (si se conoce)
    
    d_inf, d_sup: Limites en distancia de la corriente 
    
    C11, C22, C33: Valores de la diagonal de la matriz de covarianza de la corriente
    
    d_mean, e_dd: Distancia media y su error
    mu1_mean, mu2_mean: Movimientos propios medios en el frame de la corriente
    e_mu1, e_mu2, cov_mu: Valores de la matriz de cov de los movimientos propios en le frame de la correinte
    
    lim_unif: Valores limite de la distrubucion uniforme de los prior de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d y f
    
    nwalkers, ndim, steps: Numero de caminadores, parametros y pasos para la MCMC
    discard: Numero de pasos descartados por el burn-in
    thin: Numero de pasos cada cuanto tomar un valor
    
    q_min, q_max: Percentiles minimo y maximo
    
    cut_d_min, cut_d_max: Limites inferior y superior en la mascara de corte en distancia
    
    """
    
    tabla = 'RRLwithprobthin.fit' #Nombre tabla de datos
    # tabla = 'mix_table.fit'
    st = 'Pal5-PW19' #Nombre de la corriente
    
    printTrack = 'no'
    do_xd_model = 'no' #Calcular (yes/no) modelo de fondo
    N_inf, N_sup = 3, 13 #Numero min y (max+1) de gaussianas para el xd
    i_best_xd = 6 #Mejor numero de gaussians para modelo del fondo
    
    d_inf, d_sup = 18, 25 #Limites en distancia de la corriente

    #Matriz de covarianza intrinseca del stream (en ra, dec ? )
    C11, C22, C33 = 0.05**2, 0.05**2, 0.2**2  #mas/yr, mas/yr, kpc
    
    #Priors
    ra_mean, dec_mean = 229.022, -0.112
    d_mean, e_dd = 23.6, 0.8
    # mu1_mean, mu2_mean = 3.78307899, 0.71613004 #o 3.758023767746698, 0.7343094450236292 si transformo mu_ra y mu_dec del paper
    mura_mean, mudec_mean = -2.728, -2.687
    e_mura, e_mudec, rho_mu = 0.022, 0.025, -0.39 #Estos son en (alpha, delta), no tendría que transformar a los errores en phi1 y phi2?
    cov_mu = rho_mu*e_mura*e_mudec #rho_xy = sigma_xy/(sigma_x*sigma_y)

    lim_unif = [(-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-20, 15), (-20, 15), (-20, 15), (0, 1)]
    # lim_unif = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-20, 15), (-20, 15), (-20, 15), (0.004, 0.012)]

    #MCMC
    nwalkers, ndim, steps = 104, 13, 2**17#2**12
    burn_in, thin = 2**10, 2200 #50, 10 #2**10, 2200
    
    #quantiles
    q_min, q_max = 5, 95
    
    #limites de mascara de corte en distancia
    cut_d_min = 0.
    cut_d_max = 35.
    
    
    return tabla, st, printTrack, do_xd_model, N_inf, N_sup, i_best_xd, d_inf, d_sup, C11, C22, C33, d_mean, e_dd, ra_mean, dec_mean, mura_mean, mudec_mean, e_mura, e_mudec, cov_mu, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_min, q_max, cut_d_min, cut_d_max


