import numpy as np
from scipy.stats import multivariate_normal
from scipy.stats import norm


#Defino log-likelihood del stream

# def lnlike_st_viejo(theta_st):
#     """
#     Dado los parametros del modelo (theta_st) y las variables globales phi1 y C, devuelve el likelihood de las estrellas para la corriente
#     theta_st = a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d 
#     """
#     a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d = theta_st

#     model_mu1 = a_mu1 + b_mu1*(phi1.value-x_mu1) + c_mu1*(phi1.value-x_mu1)**2
#     model_mu2 = a_mu2 + b_mu2*(phi1.value-x_mu2) + c_mu2*(phi1.value-x_mu2)**2
#     model_d = a_d + b_d*(phi1.value-x_d) + c_d*(phi1.value-x_d)**2
#     model = np.array([model_mu1, model_mu2, model_d])

#     return np.diagonal(-0.5 *(np.matmul( np.matmul((y - model).T , np.linalg.inv(C) ) , (y - model) ) + np.log((2*np.pi)**y.shape[0] * np.linalg.det(C))))


def lnlike_st(theta):
    """
    Dado los parametros del modelo (theta_st) y las variables globales phi1 y C, devuelve el likelihood de las estrellas para la corriente
    
    Input:
    theta = theta_st, f
    theta_st = a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d 
    """
    a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d, _ = theta

    model_mu1 = a_mu1 + b_mu1*(phi1.value-x_mu1) + c_mu1*(phi1.value-x_mu1)**2
    model_mu2 = a_mu2 + b_mu2*(phi1.value-x_mu2) + c_mu2*(phi1.value-x_mu2)**2
    model_d = a_d + b_d*(phi1.value-x_d) + c_d*(phi1.value-x_d)**2
    model = np.array([model_mu1, model_mu2, model_d])
    
    ll_st = np.array([multivariate_normal.logpdf(y.T[i], mean=model.T[i], cov=C_tot[i]) for i in range(phi1.size)])
    
    return ll_st

#Defino log-likelihood 
def ln_likelihood(theta):
    """Dado los parametros del modelo (theta_st), el peso f y las variables globales phi1, C y el ln_likelihood del fondo (ll_bgn), devuelve el likelihood total de las estrellas, el ln likelihood de la corriente por su peso (ln(f) + ln_st) y el ln likelihhod del fondo por su peso (ln(1-f) + ll_bgn)
    
    Input:
    theta = theta_st, f
    theta_st = a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    """
    # theta_st = theta[0:12]
    f = theta[12]
    arg1 = np.log(f) + lnlike_st(theta)
    arg2 = np.log(1.-f) + ll_bgn
    # return np.sum(np.log( f * np.exp(lnlike_st(theta_st)) + (1-f) * p_bgn))
    return np.sum(np.logaddexp(arg1, arg2)), arg1, arg2


#Defino prior
def ln_unif(p, lim_inf, lim_sup):
    """ln de distribucion uniforme con limites entre lim_inf y lim_sup, para los puntos p
    """
    if lim_inf > lim_sup:
        print('Error: lim_inf > lim_sup')
    else:
        if p>lim_inf and p<lim_sup:
            return 0.0
        return -np.inf


def ln_prior2(theta, mu, sigma, d_mean, e_dd, lim_unif):
    """ln prior de los parametros del modelo (theta_st) y el peso f
    
    Inputs:
    theta = theta_st, f
    theta_st = a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    mu, sigma: Medias y matriz de covarianza para a_mu1 y a_mu2
    d_mean, e_dd: Media y varianza para a_d
    lim_unif: Limites inferior y superior (lim_inf, lim_sup) para las distribuciones uniformes de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    """
    a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d, f = theta

    lp_a12 = multivariate_normal.logpdf(np.stack((a_mu1, a_mu2), axis=-1), mean=mu, cov=sigma)
    lp_ad = norm.logpdf(a_d, loc=d_mean, scale=e_dd)

    lp_b1 = ln_unif(b_mu1, lim_unif[0][0], lim_unif[0][1])
    lp_b2 = ln_unif(b_mu2, lim_unif[1][0], lim_unif[1][1])
    lp_bd = ln_unif(b_d, lim_unif[2][0], lim_unif[2][1])

    lp_c1 = ln_unif(c_mu1, lim_unif[3][0], lim_unif[3][1])
    lp_c2 = ln_unif(c_mu2, lim_unif[4][0], lim_unif[4][1])
    lp_cd = ln_unif(c_d, lim_unif[5][0], lim_unif[5][1])

    lp_x1 = ln_unif(x_mu1, lim_unif[6][0], lim_unif[6][1])
    lp_x2 = ln_unif(x_mu2, lim_unif[7][0], lim_unif[7][1])

    lp_f = ln_unif(f, lim_unif[9][0], lim_unif[9][1])
    
#     lp_b1 = ln_unif(b_mu1, lim_unif[0], lim_unif[1])
#     lp_b2 = ln_unif(b_mu2, lim_unif[2], lim_unif[3])
#     lp_bd = ln_unif(b_d, lim_unif[4], lim_unif[5])

#     lp_c1 = ln_unif(c_mu1, lim_unif[6], lim_unif[7])
#     lp_c2 = ln_unif(c_mu2, lim_unif[8], lim_unif[9])
#     lp_cd = ln_unif(c_d, lim_unif[10], lim_unif[11])

#     lp_x1 = ln_unif(x_mu1, lim_unif[12], lim_unif[13])
#     lp_x2 = ln_unif(x_mu2, lim_unif[14], lim_unif[15])
#     lp_xd = ln_unif(x_d, lim_unif[16], lim_unif[17])

#     lp_f = ln_unif(f, lim_unif[18], lim_unif[19])

    return lp_a12 + lp_ad + lp_b1 + lp_b2 + lp_bd + lp_c1 + lp_c2 + lp_cd + lp_x1 + lp_x2 + lp_xd + lp_f


def ln_prior(theta):#, mu, sigma, d_mean, e_dd, lim_unif):
    """ln prior de los parametros del modelo (theta_st) y el peso f
    
    Inputs:
    theta = theta_st, f
    theta_st = a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    
    Global variables:
    mu, sigma: Medias y matriz de covarianza para a_mu1 y a_mu2
    d_mean, e_dd: Media y varianza para a_d
    lim_unif: Limites inferior y superior (lim_inf, lim_sup) para las distribuciones uniformes de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    """
    a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d, f = theta
    
    lp_a12 = multivariate_normal.logpdf(np.stack((a_mu1, a_mu2), axis=-1), mean=mu, cov=sigma)
    lp_ad = norm.logpdf(a_d, loc=d_mean, scale=e_dd)
    
    p = theta[3:13]
    if not all(b[0] < v < b[1] for v, b in zip(p, lim_unif)):
        return -np.inf
    
    return lp_a12 + lp_ad + 0.


def prior_sample(mu, sigma, d_mean, e_dd, lim_unif, n):
    """Muestra de puntos a partir de los priors
    
    Inputs:
    mu, sigma: Medias y matriz de covarianza para a_mu1 y a_mu2
    d_mean, e_dd: Media y varianza para a_d
    lim_unif: Limites inferior y superior (lim_inf, lim_sup) para las distribuciones uniformes de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    """
    a_mu1, a_mu2 = np.random.multivariate_normal(mu, sigma, n).T
    a_d = np.random.normal(d_mean, e_dd, n)

    b_mu1 = np.random.uniform(lim_unif[0][0], lim_unif[0][1], n).T
    b_mu2 = np.random.uniform(lim_unif[1][0], lim_unif[1][1], n).T
    b_d = np.random.uniform(lim_unif[2][0], lim_unif[2][1], n).T

    c_mu1 = np.random.uniform(lim_unif[3][0], lim_unif[3][1], n).T
    c_mu2 = np.random.uniform(lim_unif[4][0], lim_unif[4][1], n).T
    c_d = np.random.uniform(lim_unif[5][0], lim_unif[5][1], n).T

    x_mu1 = np.random.uniform(lim_unif[6][0], lim_unif[6][1], n).T
    x_mu2 = np.random.uniform(lim_unif[7][0], lim_unif[7][1], n).T
    x_d = np.random.uniform(lim_unif[8][0], lim_unif[8][1], n).T

    f = np.random.uniform(lim_unif[9][0], lim_unif[9][1], n).T

    return np.stack((a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d, f), axis=-1)


#Defino posterior
def ln_posterior(theta):#, mu, sigma, d_mean, e_dd, lim_unif):
    """ Devuelve ln posterior, y el ln likelihood de la corriente por su peso (ln(f) + ln_st) y el ln likelihhod del fondo por su peso (ln(1-f) + ll_pbgn) en forma de tupla
    
    Inputs:
    theta = theta_st, f
    theta_st = a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    
    Global variables:
    mu, sigma: Mmedias y matriz de covarianza para a_mu1 y a_mu2
    d_mean, e_dd: Media y varianza para a_d
    lim_unif: Limites inferior y superior (lim_inf, lim_sup) para las distribuciones uniformes de b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d
    
    Outputs:
    ln posterior de las estrellas
    Tupla con el ln likelihood de la corriente por su peso (ln(f) + ln_st) en la primera entrada y el ln likelihood del fondo por su peso (ln(1-f) + ll_bgn) en la segunda entrada
    """
    lp = ln_prior(theta)#, mu, sigma, d_mean, e_dd, lim_unif)
    ll, arg1, arg2 = ln_likelihood(theta)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + ll, (arg1, arg2)



def ln_likelihood_prueba(theta):

    theta_st = theta[0:12]
    f = 1.
    arg1 = np.log(f) + lnlike_st(theta_st)
    arg2 = np.log(1.-f) + ll_bgn
    # return np.sum(np.log( f * np.exp(lnlike_st(theta_st)) + (1-f) * p_bgn))
    return np.sum(np.logaddexp(arg1, arg2)), arg1, arg2

def ln_posterior_prueba(theta, mu, sigma, d_mean, e_dd, lim_unif):

    lp = ln_prior(theta, mu, sigma, d_mean, e_dd, lim_unif)
    ll, arg1, arg2 = ln_likelihood_prueba(theta)
    if not np.isfinite(lp):
        return -np.inf, None
    return lp + ll, (arg1, arg2)