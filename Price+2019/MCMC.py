# from fondo import *
# from probs import *
# from init import *

import os #Avoids issues with paralellization in emcee
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime, time
import emcee
import corner


#SERIAL RUN
##sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(phi1, y, C, p_bgn, mu, sigma, d_mean, e_dd, lim_unif))
#sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior_global, args=(mu, sigma, d_mean, e_dd, lim_unif))
#start = time.time()
#sampler.run_mcmc(pos, steps, progress=True);
#end = time.time()
#serial_time = end-start
#print(serial_time)


ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

#NCPU RUN
with Pool() as pool:
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(phi1, y, C, p_bgn, mu, sigma, d_mean, e_dd, lim_unif), pool=pool)
    sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_posterior, args=(mu, sigma, d_mean, e_dd, lim_unif), pool=pool)
    start = time.time()
    pos, _, _, _sampler.run_mcmc(pos0, discard, progress=True)
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

##Guardo las posteriors
post = [None for n in range(len(flat_samples))]

for i in range(len(flat_samples)):
    theta = flat_samples[i]
    post[i] = log_posterior(theta, phi1, y, C, p_bgn, mu, sigma, d_mean, e_dd, lim_unif)
    if i%1000==0:
        print('n =', i)


theta_post['Posterior'] = post
theta_post.to_csv('theta_post.csv', index=False)

