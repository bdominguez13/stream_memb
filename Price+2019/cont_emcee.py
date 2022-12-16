import numpy as np
import pandas as pd

import datos 
import parametros
import fondo
import probs
import resultados

import os #Avoids issues with paralellization in emcee
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime, time
import emcee


tabla, st, printTrack, do_bg_model, printBIC, N_inf, N_sup, d_inf, d_sup, C11, C22, C33, d_mean, e_dd, mu1_mean, mu2_mean, e_mu1, e_mu2, cov_mu, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_min, q_max = parametros.parametros()
data, phi1, phi2, pmphi1, pmphi2, pmphi1_reflex, pmphi2_reflex, pmra, pmdec, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out, C_tot, footprint = datos.datos(tabla, st, printTrack, C11, C22, C33, d_inf, d_sup)
ll_bgn, _, _, _ = fondo.fondo(do_bg_model, printBIC, np.arange(N_inf, N_sup), pmra, pmdec, d, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out)

y = np.array([pmphi1.value, pmphi2.value, d])
mu = np.array([mu1_mean, mu2_mean])
sigma = np.array([[(e_mu1*10)**2, (cov_mu*100)], [(cov_mu*100), (e_mu2*10)**2]])

# miembro_PW = (data['Track']==1) & (data['Memb']>0.5)
# theta_true = np.array([3.740, 0.686, 22.022, 4.102e-2, -2.826e-2, 9.460e-3, -6.423e-4, 2.832e-3, -6.327e-3, -1.072, -10.954, -16.081, miembro_PW.sum()/phi1.value.size])
# a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d, f = theta_true

probs.phi1 = phi1
probs.y = y
probs.C_tot = C_tot
probs.ll_bgn = ll_bgn
probs.mu = mu
probs.sigma = sigma
probs.d_mean = d_mean
probs.e_dd = e_dd
probs.lim_unif = lim_unif


flat_samples = pd.read_csv('long_run/theta_post.csv').to_numpy()
# memb = pd.read_csv('memb.csv').to_numpy()

steps_cont = 2**19
pos_cont = flat_samples[-nwalkers:, :ndim]

dtype = [("(arg1, arg2)", object)]
with Pool() as pool:
    sampler_cont = emcee.EnsembleSampler(nwalkers, ndim, probs.ln_posterior, pool=pool, blobs_dtype=dtype)#, args=(mu, sigma, d_mean, e_dd, lim_unif))
    start = time.time()
    sampler_cont.run_mcmc(pos_cont, steps_cont, progress=True)
    end = time.time()
    multi_time = end-start 
    print('Tiempo MCMC: ', datetime.timedelta(seconds=multi_time), 'hrs')#,serial_time/multi_time)
    

flat_samples_cont = sampler_cont.get_chain(discard=0, thin=thin, flat=True)
ln_post_cont = sampler_cont.get_log_prob(discard=0, thin=thin, flat=True)
# flat_blobs = sampler_cont.get_blobs(discard=0, thin=thin, flat=True)

flat_samples_cont = np.insert(flat_samples_cont, flat_samples_cont.shape[1], np.array(ln_post_cont), axis=1)

columns = ["$a_{\mu_{\phi_1}}$", "$a_{\mu_{\phi_2}}$", "$a_d$", "$b_{\mu_{\phi_1}}$", "$b_{\mu_{\phi_2}}$", "$b_d$", "$c_{\mu_{\phi_1}}$", "$c_{\mu_{\phi_2}}$", "$c_d$", "$x_{\mu_{\phi_1}}$", "$x_{\mu_{\phi_2}}$", "$x_d$", "f", "ln_posterior"]
theta_post = pd.DataFrame(np.concatenate((flat_samples, flat_samples_cont), axis=0), columns=columns)
theta_post.to_csv('theta_post.csv', index=False)


flat_blobs = resultados.flat_blobs(np.concatenate((flat_samples, flat_samples_cont), axis=0), ll_bgn, ndim)

print('Calculando membresías')
memb_cont = resultados.memb_cont(phi1, flat_blobs)

inside10 = memb_cont > 0.1 
inside50 = memb_cont > 0.5

Memb = pd.DataFrame({'SolID': data['SolID'], 'DR2Name': data['DR2Name'], 'Memb': memb,'inside10': inside10, 'inside50': inside50})
Memb.to_csv('memb_prob.csv', index=False)