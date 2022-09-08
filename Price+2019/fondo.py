from datos import *

from astroML.density_estimation import XDGMM
from sklearn.mixture import GaussianMixture

if do_bg_model == 'yes':

    print('\nCalculando modelo de fondo y BIC')

    X = np.vstack([pmra_out, pmdec_out, d_out]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([e_pmra_out**2, e_pmdec_out**2, e_d_out**2]).T

    #Busco mejor numero de gaussianas
    def compute_XDGMM(N, max_iter=1000):
        models = [None for n in N]
        for i in range(len(N)):
            print("N =", N[i])
            models[i] = XDGMM(n_components=N[i], max_iter=max_iter)
            models[i].fit(X, Xerr)
        return models

    def compute_GaussianMixture(N, covariance_type='full', max_iter=1000):
        models = [None for n in N]
        for i in range(len(N)):
            models[i] = GaussianMixture(n_components=N[i], max_iter=max_iter, covariance_type=covariance_type)
            models[i].fit(X)
            print("GMM_{0} converge:".format(N[i]), models[i].converged_)
        return models

    #N = np.arange(3,13) #Con 1 gaussiana da error
    #N = np.arange(6,7)
    models = compute_XDGMM(N)
    models_gmm = compute_GaussianMixture(N)

    BIC = [None for n in N]
    BIC_gmm2 = [None for n in N]
    for i in range(len(N)):
        k = (N[i]-1) + np.tri(X.shape[1]).sum()*N[i] + X.shape[1]*N[i] #N_componentes = Pesos + covariaza(matiz simetrica) + medias
        BIC[i] = -2*models[i].logL(X,Xerr) + k*np.log(X.shape[0])
        BIC_gmm2[i] = -2*np.sum(models_gmm[i].score_samples(X)) + k*np.log(X.shape[0])
    BIC_gmm = [m.bic(X) for m in models_gmm]

    i_best = np.argmin(BIC)
    i_best_gmm = np.argmin(BIC_gmm)

    xdgmm_best = models[i_best]
    gmm_best = models_gmm[i_best] #Me quedo con el mejor modelo de gmm segun xd

    fig4=plt.figure(4,figsize=(8,6))
    fig4.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.14,left=0.19,right=0.97)
    ax4=fig4.add_subplot(111)
    ax4.plot(N, np.array(BIC)/X.shape[0], '--k', marker='o', lw=2, ms=6, label='BIC$_{xd}$/N')
    #ax4.plot(N, np.array(BIC_gmm)/X.shape[0], '--', c='red', marker='o', lw=2, ms=6, label='BIC$_{gmm}$/N')
    #ax4.plot(N, np.array(BIC_gmm2)/X.shape[0], '--', c='blue', marker='o', lw=1., ms=3, label='BIC2$_{gmm}$/N')
    ax4.legend()
    ax4.set_xlabel('Nº Clusters')
    ax4.set_ylabel('BIC/N')
    ax4.grid()
    fig4.savefig('BIC.png')

    p_bgn = np.exp(gmm_best.score_samples(np.vstack([pmra, pmdec, d]).T)) #Probabilidad del fondo para cada estrella n
    np.save('p_bgn.npy', p_bgn)


    #Comparo modelo del fondo con los datos
    sample = gmm_best.sample(ra_out.size)

    fig5=plt.figure(5,figsize=(10,10))
    fig5.subplots_adjust(wspace=0.4,hspace=0.3,top=0.98,bottom=0.11,left=0.14,right=0.97)
    ax5=fig5.add_subplot(221)
    ax5.scatter(pmra_out, pmdec_out, s=1, label='Obs')
    ax5.scatter(sample[0][:,0], sample[0][:,1], s=1, label='GMM')
    # ax5.legend()
    ax5.set_xlabel('$\\alpha$ (°)')
    ax5.set_ylabel('$\delta$ (°)')
    # ax5.set_xlim([-5,1])
    # ax5.set_ylim([-5,1])

    ax5=fig5.add_subplot(222)
    ax5.scatter(pmra_out, d_out, s=1, label='Obs')
    ax5.scatter(sample[0][:,0], sample[0][:,2], s=1, label='GMM')
    ax5.legend()
    ax5.set_xlabel('$\\alpha$ (°)')
    ax5.set_ylabel('$d$ (kpc)')
    # ax5.set_xlim([-5,1])
    # ax5.set_ylim([-5,1])

    ax5=fig5.add_subplot(223)
    ax5.scatter(pmdec_out, d_out, s=1, label='Obs')
    ax5.scatter(sample[0][:,1], sample[0][:,2], s=1, label='GMM')
    ax5.set_xlabel('$\delta$ (°)')
    ax5.set_ylabel('$d$ (kpc)')
    # ax5.set_xlim([-5,1])
    # ax5.set_ylim([-5,1])

    ax5=fig5.add_subplot(224)
    ax5.hist(d_out,bins=70, alpha=0.7)
    ax5.hist(sample[0][:,2],bins=70, alpha=0.7)
    ax5.set_xlim(0,40.)
    ax5.set_xlabel('$d$ (kpc)');

    fig5.savefig('bg_sample.png')

else:
    print('\nCargando p_bgn \n')
    p_bgn = np.load('p_bgn.npy')


