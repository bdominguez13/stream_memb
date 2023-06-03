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

from astropy.table import Table
from astropy.io import fits
import astropy.coordinates as ac
_ = ac.galactocentric_frame_defaults.set('v4.0') #set the default Astropy Galactocentric frame parameters to the values adopted in Astropy v4.0
import astropy.units as u
import gala.coordinates as gc
import galstreams

import os #Avoids issues with paralellization in emcee
os.environ["OMP_NUM_THREADS"] = "1"
from multiprocessing import Pool
from multiprocessing import cpu_count
import datetime, time
import emcee
import corner


Start = datetime.datetime.now()

print('Inicio: ', Start, '\n')

print('Cargando datos \n')
tabla, st, Name, do_xd_model, N_lim, i_best_xd, C_int, d_mean, e_d_mean, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_lim, d_lim, ra_lim, dec_lim = parametros.parametros()
data, phi1, phi2, pmphi1, pmphi2, pmra, pmdec, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, mu, sigma, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, pmra_pmdec_corr_out, e_d_out, C_pm_radec, C_tot, footprint, mask = datos.datos(tabla, st, Name, C_int, d_mean, d_lim, ra_lim, dec_lim)
# data, phi1, phi2, pmphi1, pmphi2, pmra, pmdec, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, mu, sigma, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, pmra_pmdec_corr_out, e_d_out, C_pm_radec, C_tot, footprint, mask = datos.datos_gaia(tabla, st, printTrack, C11, C22, C33, d_mean, ra_mean, dec_mean, mura_mean, mudec_mean, e_mura, e_mudec, cov_mu, d_inf, d_sup, d_lim, ra_lim, dec_lim)

mwsts = galstreams.MWStreams(verbose=False, implement_Off=False)

miembro_PW = (data['Track'][mask]==1) & (data['Memb'][mask]>0.5)
theta_true = np.array([3.740, 0.686, 22.022, 4.102e-2, -2.826e-2, 9.460e-3, -6.423e-4, 2.832e-3, -6.327e-3, -1.072, -10.954, -16.081, miembro_PW.sum()/footprint.sum()])

phi1_PW = phi1[miembro_PW]
phi2_PW = phi2[miembro_PW]
pmphi1_PW = pmphi1[miembro_PW]
pmphi2_PW = pmphi2[miembro_PW]
d_PW = d[miembro_PW]

#Parametros de la corriente
y = np.array([pmphi1.value, pmphi2.value, d])

#Parametros para el prior gaussiano de los movimientos propios en el frame de la corriente
e_dd = e_d_mean*5


print('\nModelo de fondo \n')
# gmm_best = fondo.fondo(i_best_xd, pmra_out, pmdec_out, d_out)#, e_pmra_out, e_pmdec_out, e_d_out)
# ll_bgn = gmm_best.score_samples(np.vstack([pmra, pmdec, d]).T) #ln_likelihood del fondo para cada estrella n
# np.save('ll_bgn_inf.npy', ll_bgn)
ll_bgn = np.load('ll_bgn_inf.npy')

print('MCMC \n')
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

columns = ["$a_{\mu_{\phi_1}}$", "$a_{\mu_{\phi_2}}$", "$a_d$", "$b_{\mu_{\phi_1}}$", "$b_{\mu_{\phi_2}}$", "$b_d$", "$c_{\mu_{\phi_1}}$", "$c_{\mu_{\phi_2}}$", "$c_d$", "$x_{\mu_{\phi_1}}$", "$x_{\mu_{\phi_2}}$", "$x_d$", "f"]
theta_post = pd.DataFrame(flat_samples, columns=columns)

print('Guardando muestras \n')

##Guardo las posteriors
ln_post = sampler.get_log_prob(discard=0, thin=thin, flat=True)
post_true, _ = probs.ln_posterior(theta_true)

theta_post['ln_posterior'] = ln_post
theta_post.to_csv('theta_post.csv', index=False)

flat_samples = np.insert(flat_samples, flat_samples.shape[1], np.array(ln_post), axis=1)


print('Cargando datos totales del archivo viejo \n ')
f = fits.open(tabla)
data = f[1].data

c = ac.ICRS(ra=data['RA_ICRS']*u.degree, dec=data['DE_ICRS']*u.degree, distance=data['Dist']*u.kpc, pm_ra_cosdec=data['pmRA']*u.mas/u.yr, pm_dec=data['pmDE']*u.mas/u.yr, radial_velocity=np.zeros(len(data['pmRA']))*u.km/u.s) 
st_coord = c.transform_to(mwsts[st].stream_frame)
    
phi1 = st_coord.phi1 #deg
phi2 = st_coord.phi2 #deg
pmphi1 = st_coord.pm_phi1_cosphi2 #mas/yr
pmphi2 = st_coord.pm_phi2 #mas/yr
d = data['Dist'] #kpc
e_d = d*0.03 #kpc

y = np.array([pmphi1.value, pmphi2.value, d])

miembro_PW = (data['Track']==1) & (data['Memb']>0.5)

phi1_PW = phi1[miembro_PW]
phi2_PW = phi2[miembro_PW]
pmphi1_PW = pmphi1[miembro_PW]
pmphi2_PW = pmphi2[miembro_PW]
d_PW = d[miembro_PW]

#Correcion por reflejo del sol (por ahora no lo uso)
c_reflex = gc.reflex_correct(c)
st_coord_reflex = c_reflex.transform_to(mwsts[st].stream_frame)

pmphi1_reflex = st_coord_reflex.pm_phi1_cosphi2 #mas/yr
pmphi2_reflex = st_coord_reflex.pm_phi2 #mas/yr

pmra = c.pm_ra_cosdec.value #data['pmRA'][mask] #mas/yr
pmdec = c.pm_dec.value #data['pmDE'][mask] #mas/yr
e_pmra = data['e_pmRA'] #mas/yr
e_pmdec = data['e_pmDE']#mas/yr
pmra_pmdec_corr = np.zeros(len(e_pmra))


C_pm = [None for n in range(len(e_pmra))]  
for i in range(len(e_pmra)):
    C_pm[i] = np.array([[e_pmra[i]**2,0],[0,e_pmdec[i]**2]])

C_pm_radec = np.array(C_pm)
C_pm = gc.transform_pm_cov(c, C_pm_radec, mwsts[st].stream_frame) #Transformo matriz cov de pm al frame del stream

C_obs = [None for n in range(len(e_pmra))] #Matriz de covarianza observacional en el frame del stream
for i in range(len(e_pmra)):
    C_obs[i] = np.zeros((3,3))
    C_obs[i][:2,:2] = C_pm[i]
    C_obs[i][2,2] = e_d[i]**2

C_tot = C_int + C_obs

#Errores en mu1 y mu2
e_pmphi1 = np.array([C_obs[i][0,0]**0.5 for i in range(len(phi1))])
e_pmphi2 = np.array([C_obs[i][1,1]**0.5 for i in range(len(phi1))])

# ll_bgn = gmm_best.score_samples(np.vstack([pmra, pmdec, d]).T) #ln_likelihood del fondo para cada estrella n
# np.save('ll_bgn_memb.npy', ll_bgn)
ll_bgn = np.load('ll_bgn_memb.npy')

skypath = np.loadtxt('pal5_extended_skypath.icrs.txt')
skypath_N = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,1]*u.deg, frame='icrs')
skypath_S = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,2]*u.deg, frame='icrs')

# Concatenate N track, S-flipped track and add first point at the end to close the polygon (needed for ADQL)
on_poly = ac.SkyCoord(ra = np.concatenate((skypath_N.ra,skypath_S.ra[::-1],skypath_N.ra[:1])),
                        dec = np.concatenate((skypath_N.dec,skypath_S.dec[::-1],skypath_N.dec[:1])),
                        unit=u.deg, frame='icrs')

field = ac.SkyCoord(ra=data['RA_ICRS']*u.deg, dec=data['DE_ICRS']*u.deg, frame='icrs')
footprint = galstreams.get_mask_in_poly_footprint(on_poly, field, stream_frame=mwsts[st].stream_frame)

print('Calculando membresias \n')
probs.phi1 = phi1
probs.y = y
probs.C_tot = C_tot
probs.ll_bgn = ll_bgn

flat_blobs = resultados.flat_blobs(flat_samples, ll_bgn, ndim) #Lo que demora inf es esta parte
memb = resultados.memb_cont(phi1, flat_blobs)

inside10 = memb > 0.1 
inside50 = memb > 0.5

Memb = pd.DataFrame({'SolID': data['SolID'], 'DR2Name': data['DR2Name'], 'Memb': memb,'inside10': inside10, 'inside50': inside50})
Memb.to_csv('memb_prob.csv', index=False)




print('Resultados: \n')

print('theta_true: \n', theta_true,'\n')
print('inside10_PW: ', (data['Memb']>0.1).sum())
print('inside50_PW: ', (data['Memb']>0.5).sum())
print('star_PW: ', miembro_PW.sum(),'\n')


n = 500
x = np.linspace(min(phi1.value), max(phi1.value), n)
theta_max, theta_50, theta_qmin, theta_qmax, quantiles_mu1, quantiles_mu2, quantiles_d = resultados.quantiles(x, flat_samples#[flat_samples[:,ndim]>-1150]
                                                                                                              , q_lim[0], q_lim[1])
                                                                                                              # ,16,84)
print('theta_max: \n', theta_max)

print('\nGuardando resultados \n')

theta_resul = pd.DataFrame(columns = ["$a_{\mu1}$", "$a_{\mu2}$", "$a_d$", "$b_{\mu1}$", "$b_{\mu2}$", "$b_d$", "$c_{\mu1}$", "$c_{\mu2}$", "$c_d$", "$x_{\mu1}$", "$x_{\mu2}$", "$x_d$", "f", "Posterior"])
theta_resul.loc[0] = theta_max
theta_resul.loc[1] = theta_50
theta_resul.loc[2] = theta_qmin
theta_resul.loc[3] = theta_qmax
theta_resul.index = ['MAP','median','{}th'.format(q_lim[0]),'{}th'.format(q_lim[1])]
theta_resul.to_csv('theta_resul.csv', index=True)



#MAP
y_mu1 = init.model(x, theta_max[0], theta_max[3], theta_max[6], theta_max[9])
y_mu2 = init.model(x, theta_max[1], theta_max[4], theta_max[7], theta_max[10])
y_d = init.model(x, theta_max[2], theta_max[5], theta_max[8], theta_max[11])

#true
true_mu1 = init.model(x, theta_true[0], theta_true[3], theta_true[6], theta_true[9])
true_mu2 = init.model(x, theta_true[1], theta_true[4], theta_true[7], theta_true[10])
true_d = init.model(x, theta_true[2], theta_true[5], theta_true[8], theta_true[11])


print('Graficando resultados \n')

inside = inside10
star = (inside50==True) & (footprint==True)

phi1_lim = (-25,18)
phi2_lim = (-3,7)

xy_mask = (phi1.value>=phi1_lim[0]) & (phi1.value<phi1_lim[1]) & (phi2.value>=phi2_lim[0]) & (phi2.value<=phi2_lim[1])


print('Inside10: ', inside.sum())
print('Inside50: ', inside50.sum())
print('Stars: ', star.sum())


def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=256):
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

cmap = truncate_colormap(plt.get_cmap('GnBu'), minval=0, maxval=0.9) #'GnBu'


parabola = False

fig=plt.figure(1,figsize=(15,10))    
fig.subplots_adjust(wspace=0.25,hspace=0.2,top=0.9,bottom=0.25,left=0.095,right=0.98)

ax=fig.add_subplot(221)
ax.scatter(phi1[~inside], phi2[~inside], s=2, c=memb[~inside], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.scatter(phi1[inside], phi2[inside], s=20, c=memb[inside], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.plot(on_poly.transform_to(mwsts[st].stream_frame).phi1, on_poly.transform_to(mwsts[st].stream_frame).phi2, ls='--', lw=1.5, color='C1')
ax.plot(phi1_t,phi2_t,'.', c='black', ms=0.1, zorder=0)
ax.plot(phi1_PW, phi2_PW,'.', c='magenta', ms=3.)#, label='PW19')#, alpha=0.4)
ax.scatter(phi1[star], phi2[star], s=150., c=memb[star], cmap=cmap, marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)#, label='Members')
ax.set_ylabel('$\phi_2$ (°)')
ax.set_xlim(phi1_lim)
ax.set_ylim(phi2_lim)
# ax.set_xlim([min(phi1.value),max(phi1.value)])
# ax.set_xlim([-.25,.25])
# ax.set_ylim([-.25,.25])


ax=fig.add_subplot(222)
ax.scatter(phi1[~inside & xy_mask], d[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.errorbar(x=phi1[inside & xy_mask], y=d[inside & xy_mask], yerr=e_d[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
ax.scatter(phi1[inside & xy_mask], d[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.fill_between(x, quantiles_d[0], quantiles_d[2], color="orange", alpha=0.3, zorder=0)#, label='$5^{th}-95^{th}$')
ax.plot(x, quantiles_d[1], color="orangered", lw=2, zorder=4)
if parabola == True:
    for i in range(flat_samples.shape[0]):
        y_d = init.model(x, flat_samples[i,2], flat_samples[i,5], flat_samples[i,8], flat_samples[i,11])
        ax.plot(x, y_d, lw=.5, color="green", zorder=1, alpha=0.01)
ax.plot(x, y_d, lw=2, color="blue", zorder=3)
ax.plot(x, true_d, lw=1, color="black", zorder=2)
ax.plot(phi1_PW, d_PW,'.', c='magenta', ms=3.)#, alpha=0.4)
ax.scatter(phi1[star], d[star], s=200., c=memb[star], cmap=cmap, marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.set_ylabel('$d$ (kpc)')
ax.set_xlim(phi1_lim)
ax.set_ylim([13,25])
# ax.set_xlim([min(phi1.value),max(phi1.value)])
# ax.set_ylim([min(d),max(d)])
# ax.set_xlim([-.25,.25])
# ax.set_ylim([20,22])

ax=fig.add_subplot(223)
ax.scatter(phi1[~inside & xy_mask], pmphi1[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.errorbar(x=phi1[inside & xy_mask], y=pmphi1.value[inside & xy_mask], yerr=e_pmphi1[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
ax.scatter(phi1[inside & xy_mask], pmphi1[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.fill_between(x, quantiles_mu1[0], quantiles_mu1[2], color="orange", alpha=0.3, zorder=0)#, label='$5^{th}-95^{th}$')
ax.plot(x, quantiles_mu1[1], color="orangered", lw=2, zorder=4)
if parabola == True:
    for i in range(flat_samples.shape[0]):
        y_mu1 = init.model(x, flat_samples[i,0], flat_samples[i,3], flat_samples[i,6], flat_samples[i,9])
        ax.plot(x, y_mu1, lw=.5, color="green", alpha=0.01, zorder=1)
ax.plot(x, y_mu1, lw=2, color="blue", zorder=3)
ax.plot(x, true_mu1, lw=1, color="black", zorder=2)
ax.plot(phi1_PW, pmphi1_PW,'.', c='magenta', ms=3.)#, label='PW19')#, alpha=0.4)
ax.scatter(phi1[star], pmphi1[star], s=200., c=memb[star], cmap=cmap, marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1)#., label='Members')
ax.set_xlabel('$\phi_1$ (°)')
ax.set_ylabel('$\mu_{\phi_1}$ ("/year)')
ax.set_xlim(phi1_lim)
ax.set_ylim([1,6])
# ax.set_xlim([min(phi1.value),max(phi1.value)])
# ax.set_ylim([min(pmphi1.value),max(pmphi1.value)])


ax=fig.add_subplot(224)
ax.scatter(phi1[~inside & xy_mask], pmphi2[~inside & xy_mask], s=2, c=memb[~inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.errorbar(x=phi1[inside & xy_mask], y=pmphi2.value[inside & xy_mask], yerr=e_pmphi2[inside & xy_mask], lw=0, elinewidth=1, color='lightgray', zorder=0)
m = ax.scatter(phi1[inside & xy_mask], pmphi2[inside & xy_mask], s=20, c=memb[inside & xy_mask], cmap=cmap, edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)
ax.plot(x, y_mu2, lw=2, color="blue", label='MAP', zorder=3)
ax.plot(x, quantiles_mu2[1], color="orangered", lw=2, label='Median', zorder=4)
if parabola == True:
    for i in range(flat_samples.shape[0]):
        y_mu2 = init.model(x, flat_samples[i,1], flat_samples[i,4], flat_samples[i,7], flat_samples[i,10])
        ax.plot(x, y_mu2, lw=.5, color="green", alpha=0.01, zorder=1)
ax.fill_between(x, quantiles_mu2[0], quantiles_mu2[2], color="orange", alpha=0.3, zorder=0, label='$5^{th}-95^{th}$ percentile')
ax.plot(x, true_mu2, lw=1, color="black", label='PW19 track', zorder=2)
ax.plot(phi1_PW, pmphi2_PW,'.', c='magenta', ms=3.)#, label='PW19')#, alpha=0.4)
ax.scatter(phi1[star], pmphi2[star], s=200., c=memb[star], cmap='GnBu', marker='*', edgecolors='gray', linewidths=0.5, vmin=0., vmax=1.)#, label='Members')
ax.set_xlabel('$\phi_1$ (°)')
ax.set_ylabel('$\mu_{\phi_2}$ ("/year)');
ax.set_xlim(phi1_lim)
ax.set_ylim([-1.5,3.5]);
# ax.set_xlim([min(phi1.value),max(phi1.value)])
# ax.set_ylim([min(pmphi2.value),max(pmphi2.value)])

cb_ax = fig.add_axes([.15, 0.11, 0.7, 0.025])
cbar = fig.colorbar(m, cax=cb_ax, ax=ax, orientation='horizontal', label='membership probability')

fig.legend(bbox_to_anchor=(0.5,0.95) , loc='center', ncol=4)

fig.savefig('resultados.png')


columns = ["$a_{\mu_{\phi_1}}$", "$a_{\mu_{\phi_2}}$", "$a_d$", "$b_{\mu_{\phi_1}}$", "$b_{\mu_{\phi_2}}$", "$b_d$", "$c_{\mu_{\phi_1}}$", "$c_{\mu_{\phi_2}}$", "$c_d$", "$x_{\mu_{\phi_1}}$", "$x_{\mu_{\phi_2}}$", "$x_d$", "f"]

fig2 = corner.corner(flat_samples[:,0:13], labels=columns, labelpad=0.25, truths=theta_true, truth_color='magenta') #green
corner.overplot_lines(fig2, theta_max[0:13], color="blue") #blue
corner.overplot_points(fig2, theta_max[0:13][None], marker="s", color="blue")
fig2.subplots_adjust(bottom=0.05,left=0.05)

fig2.savefig('corner_plot.png')



mask_post = flat_samples[:,ndim]>-np.inf

fig4=plt.figure(4,figsize=(15,10))
fig4.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.25,left=0.09,right=0.98)

ax4=fig4.add_subplot(221)
for i in range(flat_samples[mask_post].shape[0]):
    ax4.plot(np.sqrt(np.sum(flat_samples[mask_post][i,:ndim]**2)), flat_samples[mask_post][i, ndim], '.', color='green', ms=2, alpha=0.5)
ax4.plot(np.sqrt(np.sum(theta_true**2)), post_true, '.', color='black', ms=10)
# ax4.set_xlim([15,55])
ax4.set_xlabel('$||(\\theta,f)||$')
ax4.set_ylabel('Posterior')


ax4=fig4.add_subplot(222)
for i in range(flat_samples[mask_post].shape[0]):
    y_d = init.model(x, flat_samples[mask_post][i,2], flat_samples[mask_post][i,5], flat_samples[mask_post][i,8], flat_samples[mask_post][i,11])
    ax4.plot(x, y_d, lw=.5, color="green", zorder=1, alpha=0.01)

ax4.plot(x, true_d, lw=3, color="black", label='PW19', zorder=2)
ax4.set_xlim([min(phi1.value),max(phi1.value)])
# ax4.set_ylim([min(d),max(d)])
ax4.set_xlabel('$\phi_1$ (°)')
ax4.set_ylabel('d (kpc)')


ax4=fig4.add_subplot(223)
for i in range(flat_samples[mask_post].shape[0]):
    y_mu1 = init.model(x, flat_samples[mask_post][i,0], flat_samples[mask_post][i,3], flat_samples[mask_post][i,6], flat_samples[mask_post][i,9])
    ax4.plot(x, y_mu1, lw=.5, color="green", zorder=1, alpha=0.01)

ax4.plot(x, true_mu1, lw=3, color="black", label='PW19', zorder=2)
ax4.set_xlim([min(phi1.value),max(phi1.value)])
# ax4.set_ylim([min(pmphi1.value),max(pmphi1.value)])
ax4.set_xlabel('$\phi_1$ (°)')
ax4.set_ylabel('$\mu_{\phi_1}$ ("/year)')

ax4=fig4.add_subplot(224)
for i in range(flat_samples[mask_post].shape[0]):
    y_mu2 = init.model(x, flat_samples[mask_post][i,1], flat_samples[mask_post][i,4], flat_samples[mask_post][i,7], flat_samples[mask_post][i,10])
    ax4.plot(x, y_mu2, lw=.5, color="green", zorder=1, alpha=0.01)

ax4.plot(x, true_mu2, lw=3, color="black", label='PW19', zorder=2)

ax4.set_xlim([min(phi1.value),max(phi1.value)])
# ax4.set_ylim([min(pmphi2.value),max(pmphi2.value)]);
ax4.set_xlabel('$\phi_1$ (°)')
ax4.set_ylabel('$\mu_{\phi_2}$ ("/year)');


fig4.savefig('parabolas.png')


steps = np.arange(1,flat_samples.shape[0]+1)
N = np.arange(ndim)
columns = ["$a_{\mu_{\phi_1}}$", "$a_{\mu_{\phi_2}}$", "$a_d$", "$b_{\mu_{\phi_1}}$", "$b_{\mu_{\phi_2}}$", "$b_d$", "$c_{\mu_{\phi_1}}$", "$c_{\mu_{\phi_2}}$", "$c_d$", "$x_{\mu_{\phi_1}}$", "$x_{\mu_{\phi_2}}$", "$x_d$", "f"]

# mask_f = flat_samples[:,ndim]>-1150

fig8=plt.figure(8,figsize=(12,ndim*3.5))
fig8.subplots_adjust(wspace=0.4,hspace=0.47,top=0.99,bottom=0.02,left=0.08,right=0.98)
for i in N:
    ax8=fig8.add_subplot(ndim,1,i+1)
    ax8.plot(steps, flat_samples[:,i], 'k.', ms=2)#, alpha=.5)
    # ax8.plot(steps[mask_f], flat_samples[mask_f][:,i], 'k.', ms=2)#, alpha=.5)
    # ax8.plot(steps[~mask_f], flat_samples[~mask_f][:,i], '.', color='red', ms=2)#, alpha=.5)
    ax8.plot(steps, theta_true[i]*np.ones(flat_samples.shape[0]), '-', color='magenta', lw=1.)
    ax8.set_title(columns[i])
    
fig8.savefig('steps.png')

End = datetime.datetime.now()
print('Final: ', End, '\n')


#Lo dejo para el final xq muchas veces da error por ser poco el largo de los datos..
# tau = sampler.get_autocorr_time()
# print('tau: ', tau)
# print('tau promedio: {}'.format(np.mean(tau)))