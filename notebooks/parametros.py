import numpy as np

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
    
    # tabla = 'RRLwithprobthin.fit' #Nombre tabla de datos
    # tabla = 'g_all.csv'
    st = 'Pal5-PW19' #Nombre de la corriente
    Name = 'Pal 5' #Nombre del cumulo en el catalogo de Vasiliev
    Name_d = 'Pal_5' #Nombre del cumulo en el catalogo de Baumgardt
    
    # printTrack = 'no'
    do_xd_model = 'no' #Calcular (yes/no) modelo de fondo
    N_lim = (3, 13) #Numero min y (max+1) de gaussianas para el xd
    N_best_xd = 6 #Mejor numero de gaussians para modelo del fondo
    
    # d_inf, d_sup = 18, 25 #Limites en distancia de la corriente
    
    width = 1.5 #Ancho del footprint

    #Matriz de covarianza intrinseca del stream (en phi1, phi2)
    C_int = np.diag([0.05, 0.05, 0.2])**2 #HACK, mas/yr, mas/yr, kpc
    
    #Priors
    # d_mean, e_d_mean = 23.6, 0.8 #Kupper+2015
    lim_unif = [(-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-20, 15), (-20, 15), (-20, 15), (0, 1)]

    # ra_mean, dec_mean = 229.022, -0.112
    # mu1_mean, mu2_mean = 3.78307899, 0.71613004 #o 3.758023767746698, 0.7343094450236292 si transformo mu_ra y mu_dec del paper
    # mura_mean, mudec_mean = -2.728, -2.687
    # e_mura, e_mudec, rho_mu = 0.022, 0.025, -0.39 #Estos son en (alpha, delta), no tendría que transformar a los errores en phi1 y phi2?
    # cov_mu = rho_mu*e_mura*e_mudec #rho_xy = sigma_xy/(sigma_x*sigma_y)
    # lim_unif = [(-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5), (-20, 15), (-20, 15), (-20, 15), (0.004, 0.012)]

    #MCMC
    nwalkers, ndim = 104, 13
    burn_in, steps, thin = 2**10, 2**17, 2200 #50, 2**12, 10 
    
    #quantiles
    q_lim = (5, 95)
    
    #limites de mascara
    d_lim = (10, 35)
    ra_lim = (215, 255)
    dec_lim = (-15, 10)
    
    
    return st, Name, do_xd_model, N_lim, N_best_xd, C_int, width, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_lim, d_lim, ra_lim, dec_lim


def parametros_Fjorm():
    """Devuelve los parametros a usar en el scritpt:
    st: Nombre de la corriente
    Name: Nombre del cumulo en el catalogo de Vasiliev
    Name_d: Nombre del cumulo en el catalogo de Baumgardt
    
    do_xd_model: Calcular (yes/no) modelo de fondo
    N_lim: Numero min y (max+1) de gaussianas para el xd
    i_best_xd: Mejor numero de gaussianas para el modelo del fondo (si se conoce)
    
    width: Ancho del footprint
    
    C_int: Valores de la diagonal de la matriz de covarianza intrinseca de la corriente
    
    lim_unif: Valores limite de la distrubucion uniforme de los prior de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d y f
    
    nwalkers, ndim, steps: Numero de caminadores, parametros y pasos para la MCMC
    discard: Numero de pasos descartados por el burn-in
    thin: Numero de pasos cada cuanto tomar un valor
    
    q_min, q_max: Percentiles minimo y maximo
    
    cut_d_min, cut_d_max: Limites inferior y superior en la mascara de corte en distancia
    
    """
    
    st = 'Fjorm-I21' #Nombre de la corriente
    Name = 'NGC 4590' #Nombre del cumulo en el catalogo de Vasiliev
    Name_d = 'NGC_4590' #Nombre del cumulo en el catalogo de Baumgardt
    
    do_xd_model = 'no' #Calcular (yes/no) modelo de fondo
    N_lim = (4, 15) #Numero min y (max+1) de gaussianas para el xd
    N_best_xd = 5 #Mejor numero de gaussians para modelo del fondo
    
    
    width = 8. #Ancho del footprint

    #Matriz de covarianza intrinseca del stream (en phi1, phi2)
    C_int = np.diag([1.5, 1.25, 1.15])**2 #std de I21, mas/yr, mas/yr, kpc
    
    #Priors
    lim_unif = [(-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-100, 100), (-75, 75), (-75, 75), (-75, 75), (0, 1)]

    #MCMC
    nwalkers, ndim = 104, 16
    burn_in, steps, thin = 2**11, 2**17, 2200 #50, 2**12, 10 
    
    #quantiles
    q_lim = (5, 95)
    
    #limites de mascara
    d_lim = (0, 20)
    ra_lim = (180,305) #(180.30401511707694, 306.7764957850894)
    dec_lim = (-70,100) #(-70.8059018281094, 100.4664055090781)
    
    phi1_lim = (-77.5,77.5)
    phi2_lim = (-20,7)
    
    
    return st, Name, Name_d, do_xd_model, N_lim, N_best_xd, C_int, width, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_lim, d_lim, ra_lim, dec_lim, phi1_lim, phi2_lim