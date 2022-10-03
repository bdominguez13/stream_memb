import numpy as np

def quantiles(x, flat_samples, q_min, q_max):
    """
    Funcion que devuelve los parametros de MAP, la mediana (percentil 50) y los percentiles q_min y q_max
    
    Inputs:
    x: Array 1D de n puntos equidistancias entre min(phi1) y max(phi1), siendo phi1 las posiciones horizontales de las estrellas en el frame de la corriente
    flat_samples: Resultado de la MCMC achatada, sin el burn-in y adelgazada, con la posterior de cada paso insertada al final
    q_min, q_max: Percentiles inf y sup
    
    Otputs:
    theta_max, theta_50, theta_qmin, theta_qmax: Parametros correspondientes al MAP, mediana y los percentiles q_min y q_max
    quantiles_mu1, quantiles_mu2, quantiles_d: Puntos de grafico del modelo para los percentiles q_min, 50 y q_max para mu1, mu2 y distancia
    
    """
    post = flat_samples[:, -1]
    
    MAP = max(post)
    theta_max = flat_samples[np.argmax(post)]
    
    
    x = x*np.ones((flat_samples.shape[0], x.size))
    
    a_mu1 = flat_samples[:,0].reshape(flat_samples.shape[0],1)
    a_mu2 = flat_samples[:,1].reshape(flat_samples.shape[0],1)
    a_d = flat_samples[:,2].reshape(flat_samples.shape[0],1)
    
    b_mu1 = flat_samples[:,3].reshape(flat_samples.shape[0],1)
    b_mu2 = flat_samples[:,4].reshape(flat_samples.shape[0],1)
    b_d = flat_samples[:,5].reshape(flat_samples.shape[0],1)

    c_mu1 = flat_samples[:,6].reshape(flat_samples.shape[0],1)
    c_mu2 = flat_samples[:,7].reshape(flat_samples.shape[0],1)
    c_d = flat_samples[:,8].reshape(flat_samples.shape[0],1)
    
    x_mu1 = (flat_samples[:,9]*np.ones((x.shape[1],flat_samples.shape[0]))).T
    x_mu2 = (flat_samples[:,10]*np.ones((x.shape[1],flat_samples.shape[0]))).T
    x_d = (flat_samples[:,11]*np.ones((x.shape[1],flat_samples.shape[0]))).T
    
    model_mu1 = a_mu1 + np.multiply(b_mu1, (x-x_mu1)) + np.multiply(c_mu1, (x-x_mu1)**2)
    model_mu2 = a_mu2 + np.multiply(b_mu2, (x-x_mu2)) + np.multiply(c_mu2, (x-x_mu2)**2)
    model_d = a_d + np.multiply(b_d, (x-x_d)) + np.multiply(c_d, (x-x_d)**2)
    
    quantiles_mu1 = np.percentile(model_mu1, [q_min, 50, q_max], axis=0)
    quantiles_mu2 = np.percentile(model_mu2, [q_min, 50, q_max], axis=0)
    quantiles_d = np.percentile(model_d, [q_min, 50, q_max], axis=0)
    
    
    p_50 = np.percentile(post, 50)
    i_50 = abs(post-p_50).argmin()
    theta_50 = flat_samples[i_50]

    #Percentiles q_min y q_max
    p_qmin = np.percentile(post, q_min)
    p_qmax = np.percentile(post, q_max)
    i_qmin = abs(post-p_qmin).argmin()
    i_qmax = abs(post-p_qmax).argmin()

    theta_qmin = flat_samples[i_qmin]
    theta_qmax = flat_samples[i_qmax]
    
    return theta_max, theta_50, theta_qmin, theta_qmax, quantiles_mu1, quantiles_mu2, quantiles_d


def memb(phi1, flat_blobs):
    """
    Calculo de la probabilidad de membresia a la corriente para las estrellas de la muestra
    
    Inputs:
    phi1: Posiciones horizontales de las estrellas en el frame de la corriente
    flat_blobs: Tupla obtenida del MCMC achatada, sin el burn-in y adelgazada, con el ln likelihood de la corriente por su peso (ln(f) + ln_st) en la primera entrada y el ln likelihhod del fondo por su peso (ln(1-f) + ll_bgn) en la segunda entrada
    """
    norm = 0.0
    post_prob = np.zeros(len(phi1))
    for i in range(len(flat_blobs)):
        ll_st, ll_bg = flat_blobs[i][0][0], flat_blobs[i][0][1]
        post_prob += np.exp(ll_st - np.logaddexp(ll_st, ll_bg))
        norm += 1
    post_prob /= norm
    
    return post_prob