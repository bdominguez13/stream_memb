from datos import *

from scipy.stats import multivariate_normal
from scipy.stats import norm


print('\nVAPs matriz cov: {} \n'.format(np.linalg.eig(sigma)[0]))

#Defino log-likelihood del stream
def log_st(theta_st, phi1, y, C):
    a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d = theta_st

    model_mu1 = a_mu1 + b_mu1*(phi1.value-x_mu1) + c_mu1*(phi1.value-x_mu1)**2
    model_mu2 = a_mu2 + b_mu2*(phi1.value-x_mu2) + c_mu2*(phi1.value-x_mu2)**2
    model_d = a_d + b_d*(phi1.value-x_d) + c_d*(phi1.value-x_d)**2
    model = np.array([model_mu1, model_mu2, model_d])

    return np.diagonal(-0.5 *(np.matmul( np.matmul((y - model).T , np.linalg.inv(C) ) , (y - model) ) + np.log((2*np.pi)**y.shape[0] * np.linalg.det(C))))

def log_st_global(theta_st):
    a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d = theta_st

    model_mu1 = a_mu1 + b_mu1*(phi1.value-x_mu1) + c_mu1*(phi1.value-x_mu1)**2
    model_mu2 = a_mu2 + b_mu2*(phi1.value-x_mu2) + c_mu2*(phi1.value-x_mu2)**2
    model_d = a_d + b_d*(phi1.value-x_d) + c_d*(phi1.value-x_d)**2
    model = np.array([model_mu1, model_mu2, model_d])

    return np.diagonal(-0.5 *(np.matmul( np.matmul((y - model).T , np.linalg.inv(C) ) , (y - model) ) + np.log((2*np.pi)**y.shape[0] * np.linalg.det(C))))


#Defino log-likelihood 
def log_likelihood(theta, phi1, y, C, p_bgn):
    theta_st = theta[0:12]
    f = theta[12]
    return np.sum(np.log( f * np.exp(log_st(theta_st, phi1, y, C)) + (1-f) * p_bgn))

def log_likelihood_global(theta):
    theta_st = theta[0:12]
    f = theta[12]
    return np.sum(np.log( f * np.exp(log_st_global(theta_st)) + (1-f) * p_bgn))


#Defino prior
def log_unif(p, lim_inf, lim_sup):
    if p>lim_inf and p<lim_sup:
        return 0.0
    return -np.inf


def log_prior(theta, mu, sigma, d_mean, e_dd, lim_unif):
    a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d, f = theta

    p_a12 = multivariate_normal.logpdf(np.stack((a_mu1, a_mu2), axis=-1), mean=mu, cov=sigma)
    p_ad = norm.logpdf(a_d, loc=d_mean, scale=e_dd)

    p_b1 = log_unif(b_mu1, lim_unif[0], lim_unif[1])
    p_b2 = log_unif(b_mu2, lim_unif[2], lim_unif[3])
    p_bd = log_unif(b_d, lim_unif[4], lim_unif[5])

    p_c1 = log_unif(c_mu1, lim_unif[6], lim_unif[7])
    p_c2 = log_unif(c_mu2, lim_unif[8], lim_unif[9])
    p_cd = log_unif(c_d, lim_unif[10], lim_unif[11])

    p_x1 = log_unif(x_mu1, lim_unif[12], lim_unif[13])
    p_x2 = log_unif(x_mu2, lim_unif[14], lim_unif[15])
    p_xd = log_unif(x_d, lim_unif[16], lim_unif[17])

    p_f = log_unif(f, lim_unif[18], lim_unif[19])

    return p_a12 + p_ad + p_b1 + p_b2 + p_bd + p_c1 + p_c2 + p_cd + p_x1 + p_x2 + p_xd + p_f

def prior_sample(mu, sigma, d_mean, e_dd, lim_unif, n):
    a_mu1, a_mu2 = np.random.multivariate_normal(mu, sigma, n).T
    a_d = np.random.normal(d_mean, e_dd, n)

    b_mu1 = np.random.uniform(lim_unif[0], lim_unif[1], n).T
    b_mu2 = np.random.uniform(lim_unif[2], lim_unif[3], n).T
    b_d = np.random.uniform(lim_unif[4], lim_unif[5], n).T

    c_mu1 = np.random.uniform(lim_unif[6], lim_unif[7], n).T
    c_mu2 = np.random.uniform(lim_unif[8], lim_unif[9], n).T
    c_d = np.random.uniform(lim_unif[10], lim_unif[11], n).T

    x_mu1 = np.random.uniform(lim_unif[12], lim_unif[13], n).T
    x_mu2 = np.random.uniform(lim_unif[14], lim_unif[15], n).T
    x_d = np.random.uniform(lim_unif[16], lim_unif[17], n).T

    f = np.random.uniform(lim_unif[18], lim_unif[19], n).T

    return np.stack((a_mu1, a_mu2, a_d, b_mu1, b_mu2, b_d, c_mu1, c_mu2, c_d, x_mu1, x_mu2, x_d, f), axis=-1)


#Defino posterior
def log_posterior(theta, phi1, y, C, p_bgn, mu, sigma, d_mean, e_dd, lim_unif):
    lp = log_prior(theta, mu, sigma, d_mean, e_dd, lim_unif)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, phi1, y, C, p_bgn)

def log_posterior_global(theta, mu, sigma, d_mean, e_dd, lim_unif):
    lp = log_prior(theta, mu, sigma, d_mean, e_dd, lim_unif)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood_global(theta)



