import numpy as np

def quantiles(x, flat_samples, q_min, q_max):
    """
    Funcion que devuelve los parametros de MAP, la mediana (percentil 50) y los percentiles q_min y q_max
    
    Inputs:
    x: n puntos equidistancias entre min(phi1) y max(phi1), siendo phi1 las posiciones horizontales de las estrellas en el frame de la corriente
    flat_samples: Resultado de la MCMC achatada, sin el burn-in y adelgazada, con la posterior de cada paso insertada al final
    q_min, q_max: Percentiles inf y sup
    
    Otputs:
    theta_max, theta_50, theta_qmin, theta_qmax: Parametros correspondientes al MAP, mediana y los percentiles q_min y q_max
    quantiles_mu1, quantiles_mu2, quantiles_d: Puntos de grafico del modelo para los percentiles q_min, 50 y q_max para mu1, mu2 y distancia
    
    """
    post = flat_samples[:, -1]
    
    MAP = max(post)
    theta_max = flat_samples[np.argmax(post)]
    
    model_mu1 = flat_samples[:,0] + flat_samples[:,3]*(x-flat_samples[:,9]) + flat_samples[:,6]*(x-flat_samples[:,9])**2
    model_mu2 = flat_samples[:,1] + flat_samples[:,4]*(x-flat_samples[:,10]) + flat_samples[:,7]*(x-flat_samples[:,10])**2
    model_d = flat_samples[:,2] + flat_samples[:,5]*(x-flat_samples[:11]) + flat_samples[:,8]*(x-flat_samples[:,11])**2
    
    quantiles_mu1 = np.percentile(model_mu1, [q_min, 50, q_max], axis=0)
    quantiles_mu2 = np.percentile(model_mu2, [q_min, 50, q_max], axis=0)
    quantiles_d = np.percentile(model_d, [q_min, 50, q_max], axis=0)
    
    p_50 = np.percentile(post, 50)
    i_50 = abs(post-p_50).argmin()
    theta_50 = flat_samples[i_50]

    #Percentiles q_min y q_max
    p_qmin = np.percentile(post, q_min)
    p_qmaz = np.percentile(post, q_max)
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
        post_prob2 += np.exp(ll_st - np.logaddexp(ll_st, ll_bg))
        norm += 1
    post_prob /= norm
    
    return post_prob