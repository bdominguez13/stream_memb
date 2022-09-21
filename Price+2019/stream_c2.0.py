import parametros
import datos
import fondo
import probs
import init
import resultados

import numpy as np
import pandas as pd
import pylab as plt
import scipy
import seaborn as sns
sns.set(style="ticks", context="poster")

# import astropy
# from astropy.io import fits
# from astropy.table import Table
# import astropy.coordinates as ac
# import astropy.units as u
# import gala.coordinates as gc
# import galstreams

# from astroML.density_estimation import XDGMM
# from sklearn.mixture import GaussianMixture

# from scipy.optimize import curve_fit

# from scipy.stats import multivariate_normal
# from scipy.stats import norm

import os #Avoids issues with paralellization in emcee
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime, time
import emcee
import corner	

global phi1, y, C, p_bgn, ll_bgn #Defino variables globales

Start = datetime.datetime.now()

print('Inicio: ', Start, '\n')

print('Cargo datos \n')
tabla, st, do_bg_model, printBIC, N_inf, N_sup, printBIC, d_inf, d_sup, C11, C22, C33, d_mean, e_dd, mu1_mean, mu2_mean, e_mu1, e_mu2, cov_mu, lim_unif, nwalkers, ndim, steps, discard, thin, q_min, q_max = parametros.parametros()

data, phi1, phi2, pmphi1, pmphi2, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out = datos.datos(tabla, st, d_inf, d_sup)

mu = np.array([mu1_mean, mu2_mean])
sigma = np.array([[(e_mu1*10)**2, (cov_mu*100)], [(cov_mu*100), (e_mu2*10)**2]]) #Matriz de covarianza del prior gaussiano de los movimientos propios en el frame de la corriente

y = np.array([pmphi1.value, pmphi2.value, d])
C = np.array([[C11, 0, 0], [0, C22, 0], [0, 0, C33]]) #Matriz de covarianza de la corriente: mov propios y distancia (fija)

#Para que funcione tengo que primero asignarle las variables globales al modulo probs
probs.phi1 = phi1
probs.y = y
probs.C = C

print('Modelo de fondo \n')
N = np.arange(N_inf, N_sup) #Vector con numero de gaussianas
p_bgn, gmm_best, BIC = fondo.fondo(do_bg_model, printBIC, N, pmra_out, pmdec_out, d_out, e_prma_out, e_pmdec_out, e_d_out)

probs.p_bgn = p_bgn

print('MCMC')
inside = (data['Track']==1)
miembro = inside & (data['Memb']>0.5)
pos0 = init.init(phi1, pmphi1, pmphi2, d, miembro, nwalkers, ndim)

#SERIAL RUN
#sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=(mu, sigma, d_mean, e_dd, lim_unif))
#start = time.time()
#sampler.run_mcmc(pos, steps, progress=True);
#end = time.time()
#serial_time = end-start
#print(serial_time)

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

#NCPU RUN
dtype = [("(arg1, arg2)", object)]
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=(mu, sigma, d_mean, e_dd, lim_unif), pool=pool, bolbs_dtype=dtype)
    start = time.time()
    pos, _, _, _ = sampler.run_mcmc(pos0, discard, progress=True)
    sampler.reset()
    sampler.run_mcmc(pos, steps, progress=True)
    end = time.time()
    multi_time = end-start 
    print('Tiempo MCMC: ', datetime.timedelta(seconds=multi_time), 'hrs')#,serial_time/multi_time)

tau = sampler.get_autocorr_time()
print('tau: ', tau)
print('tau promedio: {}'.format(np.mean(tau)))

flat_samples = sampler.get_chain(discard=0, thin=thin, flat=True)
print('Tamano muestra: {}'.format(flat_samples.shape))

columns = ["$a_{\mu1}$", "$a_{\mu2}$", "$a_d$", "$b_{\mu1}$", "$b_{\mu2}$", "$b_d$", "$c_{\mu1}$", "$c_{\mu2}$", "$c_d$", "$x_{\mu1}$", "$x_{\mu2}$", "$x_d$", "f"]
theta_post = pd.DataFrame(flat_samples, columns=columns)

fig6 = corner.corner(flat_samples, labels=columns, labelpad=0.25)
fig6.subplots_adjust(bottom=0.05,left=0.05)

fig6.savefig('corner_plot.png')


print('Guardando muestras y posteriors \n')

##Guardo las posteriors
post = sampler.get_log_prob(discard=0, thin=thin, flat=True)

theta_post['Posterior'] = post
theta_post.to_csv('theta_post.csv', index=False)
flat_samples = np.insert(flat_samples, flat_samples.shape[1], np.array(post), axis=1) 


#Maximum a Posterior
# MAP = max(post)
# theta_max = flat_samples[np.argmax(post)]

# #Median posterior
# argpost = np.argsort(post)
# medP = np.percentile(post,50)
# i_50 = abs(post-medP).argmin()
# # theta_med = flat_samples[argpost[int(flat_samples.shape[0]/2)]]
# theta_med = flat_samples[i_50]

# #Percentiles 5 y 95
# p5 = np.percentile(post,5)
# p95 = np.percentile(post,95)
# i_5 = abs(post-p5).argmin()
# i_95 = abs(post-p95).argmin()

# theta_5 = flat_samples[i_5]
# theta_95 = flat_samples[i_95]

theta_max, quantiles_mu1, quantiles_mu2, quantiles_d = resultados.quantiles(phi1, flat_samples, q_min, q_max)

==================================================================================================================


print('\nGuardando resultados \n')

theta_resul = pd.DataFrame(columns = ["$a_{\mu1}$", "$a_{\mu2}$", "$a_d$", "$b_{\mu1}$", "$b_{\mu2}$", "$b_d$", "$c_{\mu1}$", "$c_{\mu2}$", "$c_d$", "$x_{\mu1}$", "$x_{\mu2}$", "$x_d$", "f", "Posterior"])
theta_resul.loc[0] = theta_max
theta_resul.loc[1] = theta_med
theta_resul.loc[2] = theta_5
theta_resul.loc[3] = theta_95
theta_resul.index = ['MAP','median','5th','95th']
theta_resul.to_csv('theta_resul.csv', index=True)


print('Guardando membresias \n')

##Prob de membresia al stream
flat_blobs = sampler.get_blobs(discard=0, thin=thin, flat=True)
memb = resultados.memb(phi1, flat_blobs)

inside10 = memb > 0.1 
inside50 = memb > 0.5

Memb = pd.DataFrame({'SolID': data['SolID'], 'DR2Name': data['DR2Name'], 'Memb%': memb,'inside10': inside10, 'inside50': inside50})
Memb.to_csv('memb_prob.csv', index=False)


End = datetime.datetime.now()
print('Final: ', End, '\n')

