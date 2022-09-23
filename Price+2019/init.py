import numpy as np
from probs import prior_sample
from scipy.optimize import curve_fit


#Forma1
def init_prior(mu, sigma, d_mean, e_dd, lim_unif, nwalkers, ndim):
    """
    Inputs:
    mu, sigma: Medias y matriz de covarianza para a_mu1 y a_mu2
    d_mean, e_dd: Media y varianza para a_d
    lim_unif: Limites inferior y superior (lim_inf, lim_sup) para las distribuciones uniformes de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    nwalker: Numero de caminadores
    ndim: Numero de parametros libres
    
    Output
    pos0: Posiciones iniciales de los ndim parametros y nwalker caminadores a partir de los priors
    """
    pos0 = prior_sample(mu, sigma, d_mean, e_dd, lim_unif, nwalkers) + 1e-4*np.random.randn(nwalkers, ndim) #ndim parametros iniciales
    return pos0



#Forma2
def model(phi1, a, b, c, x):
    """Dado a,b,c,x devuelve una parabola "corrida" para los puntos phi1: a + b*(phi1-x) + c*(phi1-x)**2
    """
    return a + b*(phi1-x) + c*(phi1-x)**2

# inside = (data['Track']==1)
# miembro = inside & (data['Memb']>0.5)


def init_ls(phi1, pmphi1, pmphi2, d, miembro, nwalkers, ndim):
        """
    Inputs:
    phi1: Posiciones horizontales de las estrellas en el frame de la corriente
    pmphi1, pmphi2: Movimientos propios de las estrellas en el frame de la corriente
    d: Distancias de las estrellas
    miembro: Mascara con las estrellas que perteneces a la corriente segun informacion previa
    nwalker: Numero de caminadores
    ndim: Numero de parametros libres
    
    Output: 
    pos0: Posiciones iniciales de los ndim parametros y nwalker caminadores a partir de resolver minimos cuadrados para las estrellas que cumplan la mascara [miembro]
    """
    params_mu1, _ = curve_fit(model, phi1.value[miembro], pmphi1.value[miembro])
    params_mu2, _ = curve_fit(model, phi1.value[miembro], pmphi2.value[miembro])
    params_d, _ = curve_fit(model, phi1.value[miembro], d[miembro])
    
    init = np.array([params_mu1[0], params_mu2[0], params_d[0], params_mu1[1], params_mu2[1], params_d[1], params_mu1[2], params_mu2[2], params_d[2], params_mu1[3], params_mu2[3], params_d[3], miembro.sum()/phi1.value.size])
    pos0 = init*np.ones((nwalkers, ndim)) + init*1e-1*np.random.randn(nwalkers, ndim) #ndim parametros iniciales

    print('Valores iniciales: ', init)
    
    return pos0