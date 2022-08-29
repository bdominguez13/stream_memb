import numpy as np

tabla = 'RRLwithprobthin.fit' #Nombre tabla de datos
st = 'Pal5-PW19' #Nombre del stream

do_bg_model = 'no' #Calcular (yes/no) modelo de fondo
N = np.arange(6,7) #Numero de gaussianas para el xd
printBIC = 'no'

#Matriz de covarianza del stream
C = np.array([[0.05**2, 0., 0.], [0., 0.05**2, 0.], [0., 0., 0.2**2]]) #mas/yr, mas/yr, kpc

#Priors
d_mean, e_dd = 23.6, 0.8
mu = np.array([3.78307899, 0.71613004])
e_mu1, e_mu2, rho_mu = 0.022, 0.025, -0.39
cov_mu = rho_mu*e_mu1*e_mu2 #rho_xyes= sigma_xy/(sigma_x*sigma_y)
sigma = np.array([[(e_mu1*10)**2, -(cov_mu*10)**2], [-(cov_mu*10)**2, (e_mu2*10)**2]])

lim_unif = np.array([-100, 100, -100, 100, -100, 100, -100, 100, -100, 100, -100, 100, -20, 15, -20, 15, -20, 15, 0, 1])

#MCMC
nwalkers, ndim, steps = 104, 13, 2**17
discard, thin = 2**10, 2200


