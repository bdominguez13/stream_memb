import numpy as np
import pylab as plt
import scipy
import seaborn as sns
sns.set(style="ticks", context="poster")

from astroML.density_estimation import XDGMM
from sklearn.mixture import GaussianMixture

#Busco mejor numero de gaussianas
def compute_XDGMM(N, X, Xerr, max_iter=1000):
    """
    Modelos de extreme deconvolution del numero de gaussianas dentro del array N, siendo X los datos y Xerr la matriz de covarianza para cada dato
    """
    models = [None for n in N]
    for i in range(len(N)):
        print("N =", N[i])
        models[i] = XDGMM(n_components=N[i], max_iter=max_iter)
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
    
def fondo(do_bg_model, printBIC, N, pmra, pmdec, d, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out):
    """Busca mejor numero de gaussianas a usar con un modelo de extreme decovolution (XD) para calcular de la probabilidad de que una estrella pertenezca al backgroud (p_bgn) utilizando un modelo de mezcla de gaussianas (GMM)
    
    Inputs:
    do_bg_model: Calcular (yes) modelo de fondo o cargar directamente (no) ll_bgn de ya haber sido calculada antes
    printBIC: Imprimir (yes/no) grafico del BIC en funcón del numero de gaussianas segun XD
    N: Array con el numero de gaussianas donde busca la mejor cantidad (N[i] > 1)
    pmra, pmdec, d: Movimientos propios en ar y dec y distancias de todas las estrellas
    pmra_out, pmdec_out, d_out: Movimientos propios en ar y dec y distancias de las estrellas fuera del track
    e_prma_out, e_pmdec_out, e_d_out: Errores en movimientos propios en ar y dec y distancias de las estrellas fuera del track
    
    Outputs:
    ll_bgn: Ln probabilidad de que una estrella pertenezca al backgroud
    p_bgn: Probabilidad de que una estrella pertenezca al backgroud
    gmm_best: Modelo de GMM para el mejor numero de gaussinas segun el XD (si do_bg_model == 'yes')
    BIC_xd: Array del BIC para los diferentes numeros de gaussianas segun el XD (si do_bg_model == 'yes')
    """
    if do_bg_model == 'yes':

        print('\nCalculando modelo de fondo y BIC')

        X = np.vstack([pmra_out, pmdec_out, d_out]).T
        Xerr = np.zeros(X.shape + X.shape[-1:])
        diag = np.arange(X.shape[-1])
        Xerr[:, diag, diag] = np.vstack([e_pmra_out**2, e_pmdec_out**2, e_d_out**2]).T

        #Con N=1 gaussiana da error
        models = compute_XDGMM(N, X, Xerr)
        models_gmm = compute_GaussianMixture(N, X)

        BIC_xd = [None for n in N]
        BIC_gmm2 = [None for n in N]
        for i in range(len(N)):
            k = (N[i]-1) + np.tri(X.shape[1]).sum()*N[i] + X.shape[1]*N[i] #N_componentes = Pesos + covariaza(matiz simetrica) + medias
            BIC_xd[i] = -2*models[i].logL(X,Xerr) + k*np.log(X.shape[0])
            BIC_gmm2[i] = -2*np.sum(models_gmm[i].score_samples(X)) + k*np.log(X.shape[0])
        BIC_gmm = [m.bic(X) for m in models_gmm]

        i_best_xd = np.argmin(BIC_xd)
        i_best_gmm = np.argmin(BIC_gmm)

        xdgmm_best = models[i_best_xd]
        gmm_best = models_gmm[i_best_xd] #Me quedo con el mejor modelo de gmm segun xd
        
        if printBIC == 'yes':
            fig4=plt.figure(4,figsize=(8,6))
            fig4.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.14,left=0.19,right=0.97)
            ax4=fig4.add_subplot(111)
            ax4.plot(N, np.array(BIC_xd)/X.shape[0], '--k', marker='o', lw=2, ms=6, label='BIC$_{xd}$/N')
            #ax4.plot(N, np.array(BIC_gmm)/X.shape[0], '--', c='red', marker='o', lw=2, ms=6, label='BIC$_{gmm}$/N')
            #ax4.plot(N, np.array(BIC_gmm2)/X.shape[0], '--', c='blue', marker='o', lw=1., ms=3, label='BIC2$_{gmm}$/N')
            ax4.legend()
            ax4.set_xlabel('Nº Clusters')
            ax4.set_ylabel('BIC/N')
            ax4.grid()
            fig4.savefig('BIC.png')
        
        ll_bgn = gmm_best.score_samples(np.vstack([pmra, pmdec, d]).T) #ln_likelihood del fondo para cada estrella n
        p_bgn = np.exp(gmm_best.score_samples(np.vstack([pmra, pmdec, d]).T)) #Likelihood del fondo para cada estrella n
        np.save('ll_bgn.npy', ll_bgn)
        np.save('p_bgn.npy', p_bgn)


        #Comparo modelo del fondo con los datos
        sample = gmm_best.sample(pmra_out.size)

        fig5=plt.figure(5,figsize=(12,8))
        fig5.subplots_adjust(wspace=0.35,hspace=0.34,top=0.98,bottom=0.12,left=0.12,right=0.97)
        ax5=fig5.add_subplot(221)
        ax5.scatter(pmra_out, pmdec_out, s=1, label='Obs')
        ax5.scatter(sample[0][:,0], sample[0][:,1], s=1, label='GMM')
        # ax5.legend()
        ax5.set_xlabel('$\mu_\\alpha$ (°)')
        ax5.set_ylabel('$\mu_\delta$ (°)')
        # ax5.set_xlim([-5,1])
        # ax5.set_ylim([-5,1])

        ax5=fig5.add_subplot(222)
        ax5.scatter(pmra_out, d_out, s=1, label='Obs')
        ax5.scatter(sample[0][:,0], sample[0][:,2], s=1, label='GMM')
        ax5.legend()
        ax5.set_xlabel('$\mu_\\alpha$ (°)')
        ax5.set_ylabel('$d$ (kpc)')
        # ax5.set_xlim([-5,1])
        # ax5.set_ylim([-5,1])

        ax5=fig5.add_subplot(223)
        ax5.scatter(pmdec_out, d_out, s=1, label='Obs')
        ax5.scatter(sample[0][:,1], sample[0][:,2], s=1, label='GMM')
        ax5.set_xlabel('$\mu_\delta$ (°)')
        ax5.set_ylabel('$d$ (kpc)')
        # ax5.set_xlim([-5,1])
        # ax5.set_ylim([-5,1])

        ax5=fig5.add_subplot(224)
        ax5.hist(d_out,bins=70, alpha=0.7)
        ax5.hist(sample[0][:,2],bins=70, alpha=0.7)
        ax5.set_xlim(0,60.)
        ax5.set_xlabel('$d$ (kpc)');

        fig5.savefig('bg_sample.png')
        
        return ll_bgn, p_bgn, gmm_best, BIC_xd

    else:
        print('\nCargando ll_bgn y p_bgn \n')
        ll_bgn = np.load('ll_bgn.npy')
        p_bgn = np.load('p_bgn.npy')
        
        return ll_bgn, p_bgn, None, None


