import numpy as np
import pylab as plt
import scipy
import seaborn as sns
sns.set(style="ticks", context="poster")

from astroML.density_estimation import XDGMM
from sklearnex import patch_sklearn #accelerate your Scikit-learn applications
patch_sklearn()
from sklearn.mixture import GaussianMixture

#Busco mejor numero de gaussianas
def compute_XDGMM(N, X, Xerr, tol=1e-8, max_iter=2048):
    """
    Modelos de extreme deconvolution del numero de gaussianas dentro del array N, siendo X los datos y Xerr la matriz de covarianza para cada dato
    """
    models = [None for n in N]
    for i in range(len(N)):
        print("N =", N[i])
        models[i] = XDGMM(n_components=N[i], tol=tol, max_iter=max_iter)
        models[i].fit(X, Xerr)
    return models

def compute_GaussianMixture(N, X, covariance_type='full', max_iter=1000):
    """
    Modelos de mezcla de gaussianas del numero de gaussianas dentro del array N, siendo X los datos
    """
    models = [None for n in N]
    for i in range(len(N)):
        models[i] = GaussianMixture(n_components=N[i], max_iter=max_iter, covariance_type=covariance_type)
        models[i].fit(X)
        print("GMM_{0} converge:".format(N[i]), models[i].converged_)
    return models

def XD_minBIC(N, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out):
    """Busca mejor numero de gaussianas a usar con un modelo de extreme decovolution (XD).
    
    Inputs:
    N: Array con el numero de gaussianas donde busca la mejor cantidad (N[i] > 1)
    pmra_out, pmdec_out, d_out: Movimientos propios en ar y dec y distancias de las estrellas del background
    e_prma_out, e_pmdec_out, e_d_out: Errores en movimientos propios en ar y dec y distancias de las estrellas del background
    
    Output:
    i_best_xd: Numero de gaussianas con menor BIC
    """
    
    X = np.vstack([pmra_out, pmdec_out, d_out]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([e_pmra_out**2, e_pmdec_out**2, e_d_out**2]).T

    #Con N=1 gaussiana da error
    models = compute_XDGMM(N, X, Xerr)
    
    BIC_xd = [None for n in N]
    for i in range(len(N)):
        k = (N[i]-1) + np.tri(X.shape[1]).sum()*N[i] + X.shape[1]*N[i] #N_componentes = Pesos + covariaza(matiz simetrica) + medias
        BIC_xd[i] = -2*models[i].logL(X,Xerr) + k*np.log(X.shape[0])

    i_best_xd = np.argmin(BIC_xd)
    
    
    fig4=plt.figure(4,figsize=(8,6))
    fig4.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.14,left=0.19,right=0.97)
    
    ax4=fig4.add_subplot(111)
    ax4.plot(N, np.array(BIC_xd)/X.shape[0], '--k', marker='o', lw=2, ms=6, label='BIC$_{xd}$/N')
    ax4.legend()
    ax4.set_xlabel('Nº Clusters')
    ax4.set_ylabel('BIC/N')
    ax4.grid()
    fig4.savefig('BIC.png')
    
    return i_best_xd


def fondo_xd(i_best_xd, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, pmra_pmdec_corr_out, e_d_out):
    """Calcula un modelo de mezcla de gaussianas (GMM) para las estrellas del background usando el mejor numero de gaussianas segun un modelo de extreme decovolution (XD).
    
    Inputs:
    i_best_xd: Mejor numero de gaussianas obtenido con XD
    pmra_out, pmdec_out, d_out: Movimientos propios en ar y dec y distancias de las estrellas fuera del track
    e_prma_out, e_pmdec_out, pmra_pmdec_corr_out, e_d_out: Errores en movimientos propios en ar y dec y distancias de las estrellas fuera del track
    
    Outputs:
    xdgmm_best: Modelo de XDGMM para el mejor numero de gaussianas segun el XD
    """
    
    print('\nCalculando modelo de fondo ')

    X = np.vstack([pmra_out, pmdec_out, d_out]).T
    
    C_bg = [None for n in range(len(e_pmra_out))]  
    for i in range(len(e_pmra_out)):
        C_bg[i] = np.array([[e_pmra_out[i]**2, e_pmra_out[i]*e_pmdec_out[i]*pmra_pmdec_corr_out[i]], 
                            [e_pmra_out[i]*e_pmdec_out[i]*pmra_pmdec_corr_out[i] ,e_pmdec_out[i]**2]])
    Xerr = np.zeros((len(e_pmra_out),3,3))
    Xerr[:,:2,:2] = C_bg
    Xerr[:,2,2] = e_d_out**2
    
    N = np.array([i_best_xd])
    models_xd = compute_XDGMM(N, X, Xerr)
    xdgmm_best = models_xd[0]
    
    return xdgmm_best


def fondo(i_best_xd, pmra_out, pmdec_out, d_out):#, e_pmra_out, e_pmdec_out, e_d_out):
    """Calcula un modelo de mezcla de gaussianas (GMM) para las estrellas del background usando el mejor numero de gaussianas segun un modelo de extreme decovolution (XD).
    
    Inputs:
    i_best_xd: Mejor numero de gaussianas obtenido con XD
    pmra_out, pmdec_out, d_out: Movimientos propios en ar y dec y distancias de las estrellas fuera del track
    e_prma_out, e_pmdec_out, e_d_out: Errores en movimientos propios en ar y dec y distancias de las estrellas fuera del track
    
    Outputs:
    gmm_best: Modelo de GMM para el mejor numero de gaussianas segun el XD
    """
    
    print('\nCalculando modelo de fondo ')

    X = np.vstack([pmra_out, pmdec_out, d_out]).T
    # Xerr = np.zeros(X.shape + X.shape[-1:])
    # diag = np.arange(X.shape[-1])
    # Xerr[:, diag, diag] = np.vstack([e_pmra_out**2, e_pmdec_out**2, e_d_out**2]).T

    N = np.array([i_best_xd])
    models_gmm = compute_GaussianMixture(N, X)
    gmm_best = models_gmm[0] #Me quedo con el mejor modelo de gmm segun xd
    
#     gmm_name = 'gmm_bg'
#     np.save(gmm_name + '_weights', gmm_best.weights_, allow_pickle=False)
#     np.save(gmm_name + '_means', gmm_best.means_, allow_pickle=False)
#     np.save(gmm_name + '_covariances', gmm_best.covariances_, allow_pickle=False)
    
    #Comparo modelo del fondo con los datos
#     sample = gmm_best.sample(pmra_out.size)

#     fig5=plt.figure(5,figsize=(12,8))
#     fig5.subplots_adjust(wspace=0.35,hspace=0.34,top=0.98,bottom=0.12,left=0.12,right=0.97)
#     ax5=fig5.add_subplot(221)
#     ax5.scatter(pmra_out, pmdec_out, s=1, label='Obs')
#     ax5.scatter(sample[0][:,0], sample[0][:,1], s=1, label='GMM')
#     # ax5.legend()
#     ax5.set_xlabel('$\mu_\\alpha$ (°)')
#     ax5.set_ylabel('$\mu_\delta$ (°)')
#     # ax5.set_xlim([-5,1])
#     # ax5.set_ylim([-5,1])

#     ax5=fig5.add_subplot(222)
#     ax5.scatter(pmra_out, d_out, s=1, label='Obs')
#     ax5.scatter(sample[0][:,0], sample[0][:,2], s=1, label='GMM')
#     ax5.legend()
#     ax5.set_xlabel('$\mu_\\alpha$ (°)')
#     ax5.set_ylabel('$d$ (kpc)')
#     # ax5.set_xlim([-5,1])
#     # ax5.set_ylim([-5,1])

#     ax5=fig5.add_subplot(223)
#     ax5.scatter(pmdec_out, d_out, s=1, label='Obs')
#     ax5.scatter(sample[0][:,1], sample[0][:,2], s=1, label='GMM')
#     ax5.set_xlabel('$\mu_\delta$ (°)')
#     ax5.set_ylabel('$d$ (kpc)')
#     # ax5.set_xlim([-5,1])
#     # ax5.set_ylim([-5,1])

#     ax5=fig5.add_subplot(224)
#     ax5.hist(d_out,bins=45, alpha=0.5)
#     ax5.hist(sample[0][:,2],bins=45, alpha=0.5)
#     ax5.set_xlim(0,45.)
#     ax5.set_xlabel('$d$ (kpc)');

#     fig5.savefig('bg_sample.png')
        
    return gmm_best


