import numpy as np

def quantiles(phi1, flat_samples, q_min, q_max):
    post = flat_samples[:, -1]
    MAP = max(post)
    theta_max = flat_samples[np.argmax(post)]
    
    model_mu1 = flat_samples[:,0] + flat_samples[:,3]*(phi1.value-flat_samples[:,9]) + flat_samples[:,6]*(phi1.value-flat_samples[:,9])**2
    model_mu2 = flat_samples[:,1] + flat_samples[:,4]*(phi1.value-flat_samples[:,10]) + flat_samples[:,7]*(phi1.value-flat_samples[:,10])**2
    model_d = flat_samples[:,2] + flat_samples[:,5]*(phi1.value-flat_samples[:11]) + flat_samples[:,8]*(phi1.value-flat_samples[:,11])**2
    
    quantiles_mu1 = np.percentile(model_mu1, [q_min, 50, q_max], axis=0)
    quantiles_mu2 = np.percentile(model_mu2, [q_min, 50, q_max], axis=0)
    quantiles_d = np.percentile(model_d, [q_min, 50, q_max], axis=0)
    
    return theta_max, quantiles_mu1, quantiles_mu2, quantiles_d


def memb(phi1, flat_blobs):

    norm = 0.0
    post_prob = np.zeros(len(phi1))
    for i in range(len(flat_blobs)):
        ll_st, ll_bg = flat_blobs[i][0][0], flat_blobs[i][0][1]
        post_prob2 += np.exp(ll_st - np.logaddexp(ll_st, ll_bg))
        norm += 1
    post_prob /= norm
    
    return post_prob