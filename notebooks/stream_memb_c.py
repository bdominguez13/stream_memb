import parametros
import datos
import fondo
import probs
import init
import resultados

import numpy as np
import pandas as pd
import scipy
import pylab as plt
import matplotlib as mpl
import seaborn as sns
sns.set(style="ticks", context="poster")

# from sklearnex import patch_sklearn #accelerate your Scikit-learn applications
# patch_sklearn()
from sklearn.mixture import GaussianMixture

import astropy
from astropy.io import fits
from astropy.table import Table
import astropy.coordinates as ac
_ = ac.galactocentric_frame_defaults.set('v4.0') #set the default Astropy Galactocentric frame parameters to the values adopted in Astropy v4.0
import astropy.units as u
import gala.coordinates as gc
import galstreams
from pyia import GaiaData

import os #Avoids issues with paralellization in emcee
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime, time
import emcee
import corner


Start = datetime.datetime.now()

print('Inicio: ', Start, '\n')

st, Name, Name_d, do_xd_model, N_lim, N_best_xd, C_int, width, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_lim, d_lim, ra_lim, dec_lim = parametros.parametros_Fjorm()

data, phi1_t, phi2_t, pmphi1_t, pmphi2_t, d_t, phi1, phi2, pmphi1, pmphi2, pmra, pmdec, d, C_pm_radec, e_d, pmphi1_reflex, pmphi2_reflex, mu, sigma, d_mean, e_d_mean, C_tot, footprint, mask = datos.datos_gaiaDR3(st, Name, Name_d, width, C_int, d_lim, ra_lim, dec_lim)

#Parametros de la corriente
y = np.array([pmphi1.value, pmphi2.value, d])

#Parametros para el prior gaussiano de los movimientos propios en el frame de la corriente
e_dd = e_d_mean*5

#Probabilidad del fondo
pmra_out, pmdec_out, d_out = pmra[~footprint], pmdec[~footprint], d[~footprint]

gmm_best = fondo.fondo(N_best_xd, pmra_out, pmdec_out, d_out)#, e_pmra_out, e_pmdec_out, e_d_out)
gmm_name = 'gmm_bg'
np.save(gmm_name + '_weights', gmm_best.weights_, allow_pickle=False)
np.save(gmm_name + '_means', gmm_best.means_, allow_pickle=False)
np.save(gmm_name + '_covariances', gmm_best.covariances_, allow_pickle=False)

ll_bgn = gmm_best.score_samples(np.vstack([pmra, pmdec, d]).T) #ln_likelihood del fondo para cada estrella n


#EMCEE
# steps = 2**14
# thin = 225
# 17 2200
# 16 1100
# 15 550
# 14 225
# 13 112
# 12 56

emcee_mask = footprint

probs.phi1 = phi1[emcee_mask]
probs.y = y[:,emcee_mask]
probs.C_tot = C_tot[emcee_mask]
probs.ll_bgn = ll_bgn[emcee_mask]

probs.mu = mu
probs.sigma = sigma
probs.d_mean = d_mean
probs.e_dd = e_dd
probs.lim_unif = lim_unif
probs.ndim = ndim


ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

pos0, _ = init.init_ls(phi1, pmphi1, pmphi2, d, footprint, nwalkers, ndim) #Inicializo haciendo minimos cuadrados con las estrellas que ya se que son miembros segun PW2019


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


flat_samples = sampler.get_chain(discard=0, thin=thin, flat=True)
print('Tamano muestra: {}'.format(flat_samples.shape))

columns = ["$a_{\mu_{\phi_1}}$", "$a_{\mu_{\phi_2}}$", "$a_d$", "$b_{\mu_{\phi_1}}$", "$b_{\mu_{\phi_2}}$", "$b_d$", "$c_{\mu_{\phi_1}}$", "$c_{\mu_{\phi_2}}$", "$c_d$", "$d_{\mu_{\phi_1}}$", "$d_{\mu_{\phi_2}}$", "$d_d$", "$x_{\mu_{\phi_1}}$", "$x_{\mu_{\phi_2}}$", "$x_d$", "f"]
theta_post = pd.DataFrame(flat_samples, columns=columns)



print('Guardando muestras \n')

#Guardo las posteriors
ln_post = sampler.get_log_prob(discard=0, thin=thin, flat=True)

theta_post['ln_posterior'] = ln_post
theta_post.to_csv('theta_post.csv', index=False)

flat_samples = np.insert(flat_samples, flat_samples.shape[1], np.array(ln_post), axis=1)


print('Calculando membresias \n')
#Calculo membresias
probs.phi1 = phi1
probs.y = y
probs.C_tot = C_tot
probs.ll_bgn = ll_bgn

# mask_post = flat_samples[:,ndim] > -np.inf

flat_blobs = resultados.flat_blobs(flat_samples, ll_bgn, ndim) #Lo que demora inf es esta parte
memb = resultados.memb_cont(phi1, flat_blobs)

inside10 = memb > 0.1 
inside50 = memb > 0.5

Memb = pd.DataFrame({'SolID': data['solution_id'], 'DR3Name': data['source_id_dr3_0'], 'Memb': memb,'inside10': inside10, 'inside50': inside50})
Memb.to_csv('memb_prob.csv', index=False)


#Calculo percentiles
n = 500
x = np.linspace(min(phi1.value), max(phi1.value), n)
theta_max, theta_50, theta_qmin, theta_qmax, quantiles_mu1, quantiles_mu2, quantiles_d = resultados.quantiles(x, flat_samples, q_lim[0], q_lim[1])

print('theta_max: \n', theta_max)

#MAP parabola
y_mu1 = init.model(x, theta_max[0], theta_max[3], theta_max[6], theta_max[9], theta_max[12])
y_mu2 = init.model(x, theta_max[1], theta_max[4], theta_max[7], theta_max[10], theta_max[13])
y_d = init.model(x, theta_max[2], theta_max[5], theta_max[8], theta_max[11], theta_max[14])


print('Graficando resultados \n')

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap
cmap = truncate_colormap(plt.get_cmap('GnBu'), minval=0, maxval=0.9) 


inside = inside10
star = (inside50==True) & (footprint==True)

phi1_lim = (-75,75)
phi2_lim = (-15,5)
xy_mask = (phi1.value>=phi1_lim[0]) & (phi1.value<phi1_lim[1]) & (phi2.value>=phi2_lim[0]) & (phi2.value<=phi2_lim[1])

print('Inside10: ', inside.sum())
print('Inside50: ', inside50.sum())
print('Stars: ', star.sum())


mwsts = galstreams.MWStreams(verbose=False, implement_Off=True)
on_poly = mwsts[st].create_sky_polygon_footprint_from_track(width=width*u.deg, phi2_offset=0.*u.deg)


#Errores en pmphi, y d
C_obs = C_tot - C_int
e_pmphi1 = np.array([C_obs[i][0,0]**0.5 for i in range(len(phi1))])
e_pmphi2 = np.array([C_obs[i][1,1]**0.5 for i in range(len(phi1))])
e_d = np.array([C_obs[i][2,2]**0.5 for i in range(len(phi1))])



parabola = True

fig=plt.figure(1,figsize=(15,10))    
fig.subplots_adjust(wspace=0.25,hspace=0.2,top=0.9,bottom=0.25,left=0.095,right=0.98)

ax=fig.add_subplot(221)
ax.scatter(phi1[~inside], phi2[~inside], s=2, c=memb[~inside], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.scatter(phi1[inside], phi2[inside], s=20, c=memb[inside], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.quiver(phi1.value[inside], phi2.value[inside], pmphi1_reflex.value[inside], pmphi2_reflex.value[inside], color='gray', width=0.003, headwidth=5, headlength=6.5, headaxislength=4, alpha=.5, scale=30)
ax.plot(on_poly.transform_to(mwsts[st].stream_frame).phi1, on_poly.transform_to(mwsts[st].stream_frame).phi2, ls='--', lw=1.5, color='C1')
ax.plot(phi1_t,phi2_t,'-', c='black', lw=1.5, zorder=2)
ax.scatter(phi1[star], phi2[star], s=150., c=memb[star], cmap=cmap, marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)#, label='Members')
ax.set_ylabel('$\phi_2$ (°)')
ax.set_xlim(phi1_lim)
ax.set_ylim(phi2_lim)


ax=fig.add_subplot(222)
ax.scatter(phi1[~inside & xy_mask], d[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.errorbar(x=phi1[inside & xy_mask], y=d[inside & xy_mask], yerr=e_d[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
ax.scatter(phi1[inside & xy_mask], d[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.fill_between(x, quantiles_d[0], quantiles_d[2], color="orange", alpha=0.3, zorder=0)#, label='$5^{th}-95^{th}$')
ax.plot(x, quantiles_d[1], color="orangered", lw=2, zorder=4)
ax.plot(x, y_d, lw=2, color="blue", zorder=3)
ax.plot(phi1_t,d_t,'-', c='black', lw=1.5, zorder=2)
if parabola == True:
    for i in range(flat_samples.shape[0]):
        y_d = init.model(x, flat_samples[i,2], flat_samples[i,5], flat_samples[i,8], flat_samples[i,11], flat_samples[i,14])
        ax.plot(x, y_d, lw=.5, color="green", zorder=1, alpha=0.01)
ax.scatter(phi1[star], d[star], s=200., c=memb[star], cmap=cmap, marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1., zorder=5)
ax.set_ylabel('$d$ (kpc)')
ax.set_xlim(phi1_lim)
ax.set_ylim([0,20])


ax=fig.add_subplot(223)
ax.scatter(phi1[~inside & xy_mask], pmphi1[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.errorbar(x=phi1[inside & xy_mask], y=pmphi1.value[inside & xy_mask], yerr=e_pmphi1[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
ax.scatter(phi1[inside & xy_mask], pmphi1[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.fill_between(x, quantiles_mu1[0], quantiles_mu1[2], color="orange", alpha=0.3, zorder=0)#, label='$5^{th}-95^{th}$')
ax.plot(x, quantiles_mu1[1], color="orangered", lw=2, zorder=4)
ax.plot(x, y_mu1, lw=2, color="blue", zorder=3)
ax.plot(phi1_t,pmphi1_t,'-', c='black', lw=1.5, zorder=2)
if parabola == True:
    for i in range(flat_samples.shape[0]):
        y_mu1 = init.model(x, flat_samples[i,0], flat_samples[i,3], flat_samples[i,6], flat_samples[i,9], flat_samples[i,12])
        ax.plot(x, y_mu1, lw=.5, color="green", alpha=0.01, zorder=1)
ax.scatter(phi1[star], pmphi1[star], s=200., c=memb[star], cmap=cmap, marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1, zorder=5)#., label='Members')
ax.set_xlabel('$\phi_1$ (°)')
ax.set_ylabel('$\mu_{\phi_1}$ (mas/yr)')
ax.set_xlim(phi1_lim)
ax.set_ylim([-6,6])


ax=fig.add_subplot(224)
ax.scatter(phi1[~inside & xy_mask], pmphi2[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.errorbar(x=phi1[inside & xy_mask], y=pmphi2.value[inside & xy_mask], yerr=e_pmphi2[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
m = ax.scatter(phi1[inside & xy_mask], pmphi2[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.plot(x, quantiles_mu2[1], color="orangered", lw=2, label='Median', zorder=4)
ax.plot(x, y_mu2, lw=2, color="blue", label='MAP', zorder=3)
ax.plot(phi1_t,pmphi2_t,'-', c='black', lw=1.5, zorder=2)
if parabola == True:
    for i in range(flat_samples.shape[0]):
        y_mu2 = init.model(x, flat_samples[i,1], flat_samples[i,4], flat_samples[i,7], flat_samples[i,10], flat_samples[i,13])
        ax.plot(x, y_mu2, lw=.5, color="green", alpha=0.01, zorder=1)
ax.fill_between(x, quantiles_mu2[0], quantiles_mu2[2], color="orange", alpha=0.3, zorder=0, label='$5^{th}-95^{th}$ percentile')
ax.scatter(phi1[star], pmphi2[star], s=200., c=memb[star], cmap=cmap, marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1., zorder=5)#, label='Members')
ax.set_xlabel('$\phi_1$ (°)')
ax.set_ylabel('$\mu_{\phi_2}$ (mas/yr)');
ax.set_xlim(phi1_lim)
ax.set_ylim([-1,5]);

cb_ax = fig.add_axes([.15, 0.11, 0.7, 0.025])
cbar = fig.colorbar(m, cax=cb_ax, ax=ax, orientation='horizontal', label='membership probability')

fig.legend(bbox_to_anchor=(0.5,0.95) , loc='center', ncol=4);

fig.savefig('resultados.png')




fig2 = corner.corner(flat_samples[:,0:ndim], labels=columns, labelpad=0.25)#, truths=theta_true, truth_color='magenta') #green
corner.overplot_lines(fig2, theta_max[0:ndim], color="blue") #blue
corner.overplot_points(fig2, theta_max[0:ndim][None], marker="s", color="blue")
fig2.subplots_adjust(bottom=0.05,left=0.05)

fig2.savefig('corner_plot.png')



steps = np.arange(1,flat_samples.shape[0]+1)
N = np.arange(ndim)
fig3=plt.figure(8,figsize=(12,ndim*3.5))
fig3.subplots_adjust(wspace=0.4,hspace=0.47,top=0.99,bottom=0.02,left=0.08,right=0.98)
for i in N:
    ax3=fig3.add_subplot(ndim,1,i+1)
    ax3.plot(steps, flat_samples[:,i], 'k.', ms=2)#, alpha=.5)
    # ax3.plot(steps[~mask_post], flat_samples[~mask_post][:,i], '.', color='red', ms=2)#, alpha=.5)
    # ax3.plot(steps, theta_true[i]*np.ones(flat_samples.shape[0]), '-', color='magenta', lw=1.)
    ax3.set_title(columns[i])
    
fig3.savefig('steps.png')    


y_mu1 = init.model(x, theta_max[0], theta_max[3], theta_max[6], theta_max[9], theta_max[12])
y_mu2 = init.model(x, theta_max[1], theta_max[4], theta_max[7], theta_max[10], theta_max[13])
y_d = init.model(x, theta_max[2], theta_max[5], theta_max[8], theta_max[11], theta_max[14])


fig4=plt.figure(4,figsize=(15,10))
fig4.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.25,left=0.09,right=0.98)

ax4=fig4.add_subplot(221)
for i in range(flat_samples.shape[0]):
    ax4.plot(np.sqrt(np.sum(flat_samples[i,:ndim]**2)), flat_samples[i, ndim], '.', color='green', ms=2, alpha=0.5)
ax4.set_xlabel('$||(\\theta,f)||$')
ax4.set_ylabel('Posterior')

ax4=fig4.add_subplot(222)
ax4.plot(phi1_t, d_t, lw=2, color="black", zorder=2)
ax4.plot(x, y_d, lw=2, color="blue", label='MAP', zorder=3)
for i in range(flat_samples.shape[0]):
    y_d = init.model(x, flat_samples[i,2], flat_samples[i,5], flat_samples[i,8], flat_samples[i,11], flat_samples[i,14])
    ax4.plot(x, y_d, lw=.5, color="green", zorder=1, alpha=0.01)
ax4.set_xlim([min(phi1.value),max(phi1.value)])
ax4.set_xlabel('$\phi_1$ (°)')
ax4.set_ylabel('d (kpc)')

ax4=fig4.add_subplot(223)
ax4.plot(phi1_t, pmphi1_t, lw=2, color="black", zorder=2)
ax4.plot(x, y_mu1, lw=2, color="blue", label='MAP', zorder=3)
for i in range(flat_samples.shape[0]):
    y_mu1 = init.model(x, flat_samples[i,0], flat_samples[i,3], flat_samples[i,6], flat_samples[i,9], flat_samples[i,12])
    ax4.plot(x, y_mu1, lw=.5, color="green", zorder=1, alpha=0.01)
ax4.set_xlim([min(phi1.value),max(phi1.value)])
ax4.set_xlabel('$\phi_1$ (°)')
ax4.set_ylabel('$\mu_{\phi_1}$ (mas/yr)')

ax4=fig4.add_subplot(224)
ax4.plot(phi1_t, pmphi2_t, lw=2, color="black", zorder=2)
ax4.plot(x, y_mu2, lw=2, color="blue", label='MAP', zorder=3)
for i in range(flat_samples.shape[0]):
    y_mu2 = init.model(x, flat_samples[i,1], flat_samples[i,4], flat_samples[i,7], flat_samples[i,10], flat_samples[i,13])
    ax4.plot(x, y_mu2, lw=.5, color="green", zorder=1, alpha=0.01)
ax4.set_xlim([min(phi1.value),max(phi1.value)])
ax4.set_xlabel('$\phi_1$ (°)')
ax4.set_ylabel('$\mu_{\phi_2}$ (mas/yr)');

fig4.savefig('parabolas.png')


    
theta_resul = pd.DataFrame(columns = ["$a_{\mu1}$", "$a_{\mu2}$", "$a_d$", "$b_{\mu1}$", "$b_{\mu2}$", "$b_d$", "$c_{\mu1}$", "$c_{\mu2}$", "$c_d$", "$d_{\mu1}$", "$d_{\mu2}$", "$d_d$", "$x_{\mu1}$", "$x_{\mu2}$", "$x_d$", "f", "Posterior"])
theta_resul.loc[0] = theta_max
theta_resul.loc[1] = theta_50
theta_resul.loc[2] = theta_qmin
theta_resul.loc[3] = theta_qmax
theta_resul.index = ['MAP','median','{}th'.format(q_lim[0]),'{}th'.format(q_lim[1])]
theta_resul.to_csv('theta_resul.csv', index=False)
print('theta_resul:\n', theta_resul)



End = datetime.datetime.now()
print('Final: ', End, '\n')
print(End-Start)



tau = sampler.get_autocorr_time()
print('tau: ', tau)
print('tau promedio: {}'.format(np.mean(tau)))
