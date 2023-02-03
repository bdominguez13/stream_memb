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

import os #Avoids issues with paralellization in emcee
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime, time
import emcee
import corner	


Start = datetime.datetime.now()

print('Inicio: ', Start, '\n')

tabla, st, printTrack, do_bg_model, printBIC, N_inf, N_sup, d_inf, d_sup, C11, C22, C33, d_mean, e_dd, mu1_mean, mu2_mean, e_mu1, e_mu2, cov_mu, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_min, q_max = parametros.parametros()

data, phi1, phi2, pmphi1, pmphi2, pmphi1_reflex, pmphi2_reflex, pmra, pmdec, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out, C_tot, footprint = datos.datos(tabla, st, printTrack, C11, C22, C33, d_inf, d_sup)

sgr = data['Dist'] > 40 #Creo máscara para sacar a la corriente de Sagitario de la ecuacion
miembro_PW = (data['Track'][~sgr]==1) & (data['Memb'][~sgr]>0.5)

#Parametros de la corriente
y = np.array([pmphi1.value, pmphi2.value, d])

#Parametros para el prior gaussiano de los movimientos propios en el frame de la corriente
mu = np.array([mu1_mean, mu2_mean])
sigma = np.array([[(e_mu1*10)**2, (cov_mu*100)], [(cov_mu*100), (e_mu2*10)**2]]) 
e_dd = e_dd*5

print('\nModelo de fondo \n')
N = np.arange(N_inf, N_sup) #Vector con numero de gaussianas
ll_bgn, p_bgn, gmm_best, BIC = fondo.fondo(do_bg_model, printBIC, N, pmra, pmdec, d, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out)


#Para que funcione tengo que primero asignarle las variables globales al modulo probs
probs.phi1 = phi1
probs.y = y
probs.C_tot = C_tot
probs.ll_bgn = ll_bgn
probs.p_bgn = p_bgn

probs.mu = mu
probs.sigma = sigma
probs.d_mean = d_mean
probs.e_dd = e_dd
probs.lim_unif = lim_unif


print('MCMC')
pos0, _ = init.init_ls(phi1, pmphi1, pmphi2, d, miembro_PW, nwalkers, ndim) #Inicializo haciendo minimos cuadrados con las estrellas que ya se que son miembros segun PW2019


#SERIAL RUN
# dtype = [("(arg1, arg2)", object)]
#sampler = emcee.EnsembleSampler(nwalkers, ndim, probs.ln_posterior, args=(mu, sigma, d_mean, e_dd, lim_unif), blobs_dtype=dtype)
#start = time.time()
# pos, _, _, _ = sampler.run_mcmc(pos0, burn_in, progress=True)
# sampler.reset()
# sampler.run_mcmc(pos, steps, progress=True)
#end = time.time()
#serial_time = end-start
#print(serial_time)

ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

#NCPU RUN
dtype = [("(arg1, arg2)", object)]
with Pool() as pool:
    sampler = emcee.EnsembleSampler(nwalkers, ndim, probs.ln_posterior, pool=pool, blobs_dtype=dtype)#, args=(mu, sigma, d_mean, e_dd, lim_unif))
    start = time.time()
    pos, _, _, _ = sampler.run_mcmc(pos0, burn_in, progress=True)
    sampler.reset()
    pos_final, _, _, _ = sampler.run_mcmc(pos, steps, progress=True)
    end = time.time()
    multi_time = end-start 
    print('Tiempo MCMC: ', datetime.timedelta(seconds=multi_time), 'hrs')#,serial_time/multi_time)


# np.save('pos_final.npy', pos_final)

flat_samples = sampler.get_chain(discard=0, thin=thin, flat=True)
print('Tamano muestra: {}'.format(flat_samples.shape))

columns = ["$a_{\mu_{\phi_1}}$", "$a_{\mu_{\phi_2}}$", "$a_d$", "$b_{\mu_{\phi_1}}$", "$b_{\mu_{\phi_2}}$", "$b_d$", "$c_{\mu_{\phi_1}}$", "$c_{\mu_{\phi_2}}$", "$c_d$", "$x_{\mu_{\phi_1}}$", "$x_{\mu_{\phi_2}}$", "$x_d$", "f"]
theta_post = pd.DataFrame(flat_samples, columns=columns)

print('Guardando muestras \n')

##Guardo las posteriors
ln_post = sampler.get_log_prob(discard=0, thin=thin, flat=True)

theta_post['ln_posterior'] = ln_post
theta_post.to_csv('theta_post.csv', index=False)


print('Guardando membresias \n')

##Prob de membresia al stream
flat_blobs = sampler.get_blobs(discard=0, thin=thin, flat=True)
memb = resultados.memb(phi1, flat_blobs)

inside10 = memb > 0.1 
inside50 = memb > 0.5

Memb = pd.DataFrame({'SolID': data['SolID'], 'DR2Name': data['DR2Name'], 'Memb': memb,'inside10': inside10, 'inside50': inside50})
Memb.to_csv('memb_prob.csv', index=False)



print('Guardando percentiles \n')
#MAP, median y percentiles
flat_samples = np.insert(flat_samples, flat_samples.shape[1], np.array(ln_post), axis=1)

n = 500
x = np.linspace(min(phi1.value), max(phi1.value), n)
theta_max, theta_50, theta_qmin, theta_qmax, quantiles_mu1, quantiles_mu2, quantiles_d = resultados.quantiles(x, flat_samples, q_min, q_max)


theta_true = np.array([3.740, 0.686, 22.022, 4.102e-2, -2.826e-2, 9.460e-3, -6.423e-4, 2.832e-3, -6.327e-3, -1.072, -10.954, -16.081, miembro_PW.sum()/phi1.value.size])

true_mu1 = init.model(x, theta_true[0], theta_true[3], theta_true[6], theta_true[9])
true_mu2 = init.model(x, theta_true[1], theta_true[4], theta_true[7], theta_true[10])
true_d = init.model(x, theta_true[2], theta_true[5], theta_true[8], theta_true[11])


fig6 = corner.corner(flat_samples[:,:ndim], labels=columns, labelpad=0.25, truths=theta_true)
corner.overplot_lines(fig6, theta_max[0:ndim], color="C1")
corner.overplot_points(fig6, theta_max[0:ndim][None], marker="s", color="C1")
fig6.subplots_adjust(bottom=0.05,left=0.05)
fig6.savefig('corner_plot.png')


#Median --> No lo uso, puede dar fruta: Si la distribución es prefectamente bimodal, la mediana tiene probabilidad ~0
median_mu1 = init.model(x, theta_50[0], theta_50[3], theta_50[6], theta_50[9])
median_mu2 = init.model(x, theta_50[1], theta_50[4], theta_50[7], theta_50[10])
median_d = init.model(x, theta_50[2], theta_50[5], theta_50[8], theta_50[11])

#MAP
y_mu1 = init.model(x, theta_max[0], theta_max[3], theta_max[6], theta_max[9])
y_mu2 = init.model(x, theta_max[1], theta_max[4], theta_max[7], theta_max[10])
y_d = init.model(x, theta_max[2], theta_max[5], theta_max[8], theta_max[11])


#Errores en mu1 y mu2
C_int = np.array([[C11, 0, 0], [0, C22, 0], [0, 0, C33]])
C_obs = C_tot - C_int

e_pmphi1 = np.array([C_obs[i][0,0]**0.5 for i in range(len(phi1))])
e_pmphi2 = np.array([C_obs[i][1,1]**0.5 for i in range(len(phi1))])
e_d = d*0.03

print('\nGuardando resultados \n')

columns2 = ["$a_{\mu_{\phi_1}}$", "$a_{\mu_{\phi_2}}$", "$a_d$", "$b_{\mu_{\phi_1}}$", "$b_{\mu_{\phi_2}}$", "$b_d$", "$c_{\mu_{\phi_1}}$", "$c_{\mu_{\phi_2}}$", "$c_d$", "$x_{\mu_{\phi_1}}$", "$x_{\mu_{\phi_2}}$", "$x_d$", "f", "ln_posterior"]
theta_resul = pd.DataFrame(columns = columns2)
theta_resul.loc[0] = theta_max
theta_resul.loc[1] = theta_50
theta_resul.loc[2] = theta_qmin
theta_resul.loc[3] = theta_qmax
theta_resul.index = ['MAP','median','{}th'.format(q_min),'{}th'.format(q_max)]
theta_resul.to_csv('theta_resul.csv', index=True)


print('Graficando resultados')

inside = inside10
star = (inside50==True) & (footprint==True)

xy_mask = (phi1.value>=-20.) & (phi1.value<=15.) & (phi2.value>=-3.) & (phi2.value<=5.) #Ver como automatizar estos valores

print('Inside: ', inside.sum())
print('Stars: ', star.sum())

#Parabolas del MAP y percentiles con las estrellas inside10 y estrellas miembros
fig7=plt.figure(7,figsize=(15,10))
fig7.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.25,left=0.09,right=0.98)

ax7=fig7.add_subplot(221)
ax7.scatter(phi1[~inside], phi2[~inside], s=2, c=memb[~inside], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax7.quiver(phi1.value[inside], phi2.value[inside], pmphi1_reflex.value[inside], pmphi2_reflex.value[inside], color='gray', width=0.003, headwidth=5, headlength=6.5, headaxislength=4, alpha=.5, scale=30)
m = ax7.scatter(phi1[inside], phi2[inside], s=20, c=memb[inside], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
# cb = plt.colorbar(m)
ax7.plot(phi1_t,phi2_t,'k.',ms=0.5, zorder=0)
ax7.plot(phi1[miembro_PW], phi2[miembro_PW],'.', c='black', ms=3., label='PW19')#, alpha=0.4)
ax7.scatter(phi1[star], phi2[star], s=150., c=memb[star], cmap='YlGnBu', marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1., label='Members')
ax7.set_ylabel('$\phi_2$ (°)')
ax7.set_xlim([-20,15])
ax7.set_ylim([-3,5])


ax7=fig7.add_subplot(222)
ax7.scatter(phi1[~inside & xy_mask], d[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax7.errorbar(x=phi1[inside & xy_mask], y=d[inside & xy_mask], yerr=e_d[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
m = ax7.scatter(phi1[inside & xy_mask], d[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
# cb = plt.colorbar(m)
ax7.plot(x, quantiles_d[1], color="orangered", lw=2, label='Median', zorder=0)
ax7.plot(x, y_d, lw=2, color="blue", label='MAP', zorder=1)
ax7.plot(x, true_d, lw=1, color="black", label='PW19', zorder=2)
ax7.fill_between(x, quantiles_d[0], quantiles_d[2], color="orange", alpha=0.3, label='$5^{th}-95^{th}$')
ax7.plot(phi1[miembro_PW], d[miembro_PW],'.', c='black', ms=3., label='PW19')#, alpha=0.4)
ax7.scatter(phi1[star], d[star], s=200., c=memb[star], cmap='YlGnBu', marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1., label='Members')
ax7.set_ylabel('$d$ (kpc)')
ax7.set_xlim([-20,15])
ax7.set_ylim([13,25])


ax7=fig7.add_subplot(223)
ax7.scatter(phi1[~inside & xy_mask], pmphi1[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax7.errorbar(x=phi1[inside & xy_mask], y=pmphi1.value[inside & xy_mask], yerr=e_pmphi1[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
m = ax7.scatter(phi1[inside & xy_mask], pmphi1[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
# cb = plt.colorbar(m)
ax7.plot(x, quantiles_mu1[1], color="orangered", lw=2, label='Median', zorder=0)
ax7.plot(x, y_mu1, lw=2, color="blue", label='MAP', zorder=1)
ax7.plot(x, true_mu1, lw=1, color="black", label='PW19', zorder=2)
ax7.fill_between(x, quantiles_mu1[0], quantiles_mu1[2], color="orange", alpha=0.3, label='$5^{th}-95^{th}$')
ax7.plot(phi1[miembro_PW], pmphi1[miembro_PW],'.', c='black', ms=3., label='PW19')#, alpha=0.4)
ax7.scatter(phi1[star], pmphi1[star], s=200., c=memb[star], cmap='YlGnBu', marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1., label='Members')
ax7.set_xlabel('$\phi_1$ (°)')
ax7.set_ylabel('$\mu_{\phi_1}$ ("/año)')
ax7.set_xlim([-20,15])
ax7.set_ylim([1,6])


ax7=fig7.add_subplot(224)
ax7.scatter(phi1[~inside & xy_mask], pmphi2[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax7.errorbar(x=phi1[inside & xy_mask], y=pmphi2.value[inside & xy_mask], yerr=e_pmphi2[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
m = ax7.scatter(phi1[inside & xy_mask], pmphi2[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap='YlGnBu', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
# cb = plt.colorbar(m)
ax7.plot(x, quantiles_mu2[1], color="orangered", lw=2, label='Median', zorder=0)
ax7.plot(x, y_mu2, lw=2, color="blue", label='MAP', zorder=1)
ax7.plot(x, true_mu2, lw=1, color="black", label='PW19', zorder=2)
ax7.fill_between(x, quantiles_mu2[0], quantiles_mu1[2], color="orange", alpha=0.3, label='$5^{th}-95^{th}$')
ax7.plot(phi1[miembro_PW], pmphi2[miembro_PW],'.', c='black', ms=3., label='PW19')#, alpha=0.4)
ax7.scatter(phi1[star], pmphi2[star], s=200., c=memb[star], cmap='YlGnBu', marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1., label='Members')
ax7.set_xlabel('$\phi_1$ (°)')
ax7.set_ylabel('$\mu_{\phi_2}$ ("/año)');
ax7.set_xlim([-20,15])
ax7.set_ylim([-2.5,2.5]);

cb_ax = fig7.add_axes([.15, 0.095, 0.7, 0.02])
cbar = fig7.colorbar(m, cax=cb_ax, ax=ax7, orientation='horizontal', label='probabilidad de membresía')

fig7.savefig('resultados.png')

#Valor de lo parametros en funcion de los pasos
steps = np.arange(1,flat_samples.shape[0]+1)
N = np.arange(ndim)

fig8=plt.figure(8,figsize=(12,ndim*3.5))
fig8.subplots_adjust(wspace=0.4,hspace=0.47,top=0.99,bottom=0.02,left=0.08,right=0.98)
for i in N:
    ax8=fig8.add_subplot(ndim,1,i+1)
    ax8.plot(steps, flat_samples[:,i], 'k.', ms=2)#, alpha=.5)
    ax8.plot(steps, theta_true[i]*np.ones(flat_samples.shape[0]), '-', c='orange', lw=1.)
    ax8.set_title(columns[i])

fig8.savefig('steps.png')



End = datetime.datetime.now()
print('Final: ', End, '\n')


#Lo dejo para el final xq muchas veces da error por ser poco el largo de los datos..
tau = sampler.get_autocorr_time()
print('tau: ', tau)
print('tau promedio: {}'.format(np.mean(tau)))
