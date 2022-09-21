import numpy as np
from probs import prior_sample
from scipy.optimize import curve_fit


#Forma1
def init_prior(mu, sigma, d_mean, e_dd, lim_unif, nwalkers, ndim):
    pos0 = prior_sample(mu, sigma, d_mean, e_dd, lim_unif, nwalkers) + 1e-4*np.random.randn(nwalkers, ndim) #ndim parametros iniciales
    return pos0



#Forma2
def model(phi1, a, b, c, x):
    return a + b*(phi1-x) + c*(phi1-x)**2

# inside = (data['Track']==1)
# miembro = inside & (data['Memb']>0.5)


def init_ls(phi1, pmphi1, pmphi2, d, miembro, nwalkers, ndim):
    
    params_mu1, _ = curve_fit(model, phi1.value[miembro], pmphi1.value[miembro])
    params_mu2, _ = curve_fit(model, phi1.value[miembro], pmphi2.value[miembro])
    params_d, _ = curve_fit(model, phi1.value[miembro], d[miembro])
    
    init = np.array([params_mu1[0], params_mu2[0], params_d[0], params_mu1[1], params_mu2[1], params_d[1], params_mu1[2], params_mu2[2], params_d[2], params_mu1[3], params_mu2[3], params_d[3], miembro.sum()/phi1.value.size])
    pos0 = init*np.ones((nwalkers, ndim)) + init*1e-1*np.random.randn(nwalkers, ndim) #ndim parametros iniciales

    print('Valores iniciales: ', init)
    
    return pos0