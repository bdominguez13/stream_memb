import parametros
import datos
import fondo
import datetime

import numpy as np
from xdgmm import XDGMM
import pylab as plt

import warnings
warnings.filterwarnings('ignore')

st, Name, Name_d, do_xd_model, N_lim, N_best_xd, C_int, width, lim_unif, nwalkers, ndim, steps, burn_in, thin, q_lim, d_lim, ra_lim, dec_lim = parametros.parametros_Fjorm()

data, phi1_t, phi2_t, pmphi1_t, pmphi2_t, d_t, phi1, phi2, pmphi1, pmphi2, pmra, pmdec, d, C_pm_radec, e_d, pmphi1_reflex, pmphi2_reflex, mu, sigma, d_mean, e_d_mean, C_tot, footprint, mask = datos.datos_gaiaDR3(st, Name, Name_d, width, C_int, d_lim, ra_lim, dec_lim)

pmra_out, pmdec_out, d_out = pmra[~footprint], pmdec[~footprint], d[~footprint]
C_pm_radec_out, e_d_out = C_pm_radec[~footprint], e_d[~footprint]

X = np.vstack([pmra_out, pmdec_out, d_out]).T
Xerr = np.zeros(X.shape + X.shape[-1:])
Xerr[:,:2,:2] = C_pm_radec_out
Xerr[:,2,2] = e_d_out**2

N = np.arange(N_lim[0], N_lim[1])

#modelo1
Start1 = datetime.datetime.now()

models = fondo.compute_XDGMM(N, X, Xerr)
BIC_xd = [None for n in N]
for i in range(len(N)):
    k = (N[i]-1) + np.tri(X.shape[1]).sum()*N[i] + X.shape[1]*N[i] #N_componentes = Pesos + covariaza(matiz simetrica) + medias
    BIC_xd[i] = -2*models[i].logL(X,Xerr) + k*np.log(X.shape[0])
    
N_best_xd = N[np.argmin(BIC_xd)]

End1 = datetime.datetime.now()
print('time1: ', End1-Star1,'\n')

Delta_BIC1 = np.zeros((len(N),len(N)))
for i in range(len(N)):
    for j in range(len(N)):
        Delta_BIC1[i,j] = BIC_xd[i] - BIC_xd[j]

print('BIC1:', BIC_xd,'\n')
print('N_best1:',N_best_xd,'\n')
i_best_xd1 = np.where(N == N_best_xd)[0][0]
print('Delta_BIC1[N_best]:',Delta_BIC1[:,i_best_xd1],'\n')
bla1 = np.abs(Delta_BIC1)<10
print('Delta_BIC1<10:\n',bla1)


#modelo2
Start2 = datetime.datetime.now()

xdgmm = XDGMM()
param_range = np.arange(N_lim[0],N_lim[1])
# Loop over component numbers, fitting XDGMM model and computing the BIC:
bic, optimal_n_comp, lowest_bic = xdgmm.bic_test(X, Xerr, param_range)

End2 = datetime.datetime.now()
print('time2: ', End2-Star2,'\n')

Delta_BIC2 = np.zeros((len(param_range),len(param_range)))
for i in range(len(param_range)):
    for j in range(len(param_range)):
        Delta_BIC2[i,j] = bic[i] - bic[j]


print('\n BIC2:', bic,'\n')
print('\n N_best2:', optimal_n_comp,'\n')
i_best_xd2 = np.where(param_range == optimal_n_comp)[0][0]
print('Delta_BIC2[N_best]:', Delta_BIC2[:,i_best_xd2],'\n')
bla2 = np.abs(Delta_BIC2)<10
print('Delta_BIC2<10:\n',bla2)



fig = plt.figure(1,figsize=(15,6))
fig.subplots_adjust(wspace=0.3,hspace=0.34,top=0.94,bottom=0.14,left=0.12,right=0.99)
ax = fig.add_subplot(121)
ax.plot(N, BIC_xd, '--k', marker='o', lw=2, ms=6)
ax.grid()
ax.set_xlabel('Nº Clusters')
ax.set_ylabel('BIC')
ax.set_title('modelo 1')

ax = fig.add_subplot(122)
ax.plot(param_range, bic, '--k', marker='o', lw=2, ms=6)
ax.grid()
ax.set_xlabel('Nº Clusters')
ax.set_title('modelo 2')

fig.savefig('BICs.png')