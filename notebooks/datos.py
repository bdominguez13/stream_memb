import pandas as pd
import numpy as np
import scipy
import pylab as plt
import seaborn as sns
sns.set(style="ticks", context="poster")

import astropy
from astropy.table import Table
import astropy.coordinates as ac
_ = ac.galactocentric_frame_defaults.set('v4.0') #set the default Astropy Galactocentric frame parameters to the values adopted in Astropy v4.0
import astropy.units as u
from astropy.io import fits
import gala.coordinates as gc
import galstreams
from pyia import GaiaData


def datos(tabla, st, Name, C_int, d_mean, d_lim, ra_lim, dec_lim):
    """
    Funcion que devuelve la posicion, movimientos propios y distancia de las estrellas y del track en el frame de la corriente, y los movimientos propios en ar y dec y distancia de las estrellas por fuera del track junto a sus errores
    
    Inputs:
    tabla: Nombre de la tabla donde se encuentran los datos crudos
    st: Nombre de la corriente estelar
    printTrack: Imprime (yes/no) el track en ar y dec, pm_ra y pm_dec, phi1 y phi2, phi1 y d, phi1 y pmphi1, phi1 y pmphi2
    C11, C22, C33: Valores de la matriz de covarianza intriseca del stream en movimientos propios y distancia
    ra_mean, dec_mean: Ascención recta y declinacion medios del stream
    mura_mean, mura_mean: Movimientos propios medios en ascención recta y declinacion 
    e_mura, e_mura, cov_mu: Errores de los movimientos propios medios en ascencion recta y declinacion, y el factor de correlacion entre ambos
    d_inf, d_sup: Distancia minima y maxima de la corriente
    d_lim: Limites inferior y superior en la mascara de corte en distancia

    
    Outputs:
    data: Tabla con los datos originales
    phi1, phi2: Posicion de las estrellas en el frame de la corriente
    pmphi1, pmphi2: Moviemientos propios de las estrellas en el frame de la corriente
    pmphi1_reflex, pmphi2_reflex: Moviemientos propios de las estrellas en el frame de la corriente corregidas por el reflejo solar
    pmra, pmdec, d: Movimientos propios en ascención recta y declinacion y distancias de las estrellas
    phi1_t, phi2_t, pmphi1_t, pmphi2_t: Posicion y movimientos propios del track en el frame de la corriente
    pmra_out, pmdec_out, d_out: movimientos propios y distancia de las estrellas fuera del track en ar y dec
    e_pmra_out, e_pmdec_out, e_d_out: Errores en los movimientos propios y distancia de las estrellas fuera del track en ar y dec
    C_tot: Matriz de covarianza intrinseca (fija) + observacional en el frame del stream
    footprint: Mascara con las estrellas espacialmente dentro del track
    cut_d: Corte en distancia de las estrellas que no voy a tomar en cuenta
    """

    print('\nCargando datos \n')

    ##Cargo datos
    f = fits.open(tabla)
    data = f[1].data

    print('Cargando track y transformando coordenadas \n')
    
    ##Cargo track y transformo coordenadas
    mwsts = galstreams.MWStreams(verbose=False, implement_Off=True)
    track = mwsts[st].track
    #track = gc.reflex_correct(track) #Por ahora no uso la correcion por reflejo del sol
    st_track = track.transform_to(mwsts[st].stream_frame)
    
    phi1_t = st_track.phi1
    phi2_t = st_track.phi2
    pmphi1_t = st_track.pm_phi1_cosphi2
    pmphi2_t = st_track.pm_phi2
    
    
    #mascara en datos
    mask = (data['Dist'] > d_lim[0]) & (data['Dist'] < d_lim[1]) & (data['pmRA'] != 0.) & (data['pmDE'] !=0.)# & (data['RA_ICRS'] < 250) & (data['DE_ICRS'] > -10.)

        
    c = ac.ICRS(ra=data['RA_ICRS'][mask]*u.degree, dec=data['DE_ICRS'][mask]*u.degree, distance=data['Dist'][mask]*u.kpc, pm_ra_cosdec=data['pmRA'][mask]*u.mas/u.yr, pm_dec=data['pmDE'][mask]*u.mas/u.yr, radial_velocity=np.zeros(len(data['pmRA'][mask]))*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
    st_coord = c.transform_to(mwsts[st].stream_frame)
    
    
    phi1 = st_coord.phi1 #deg
    phi2 = st_coord.phi2 #deg
    pmphi1 = st_coord.pm_phi1_cosphi2 #mas/yr
    pmphi2 = st_coord.pm_phi2 #mas/yr
    d = data['Dist'][mask] #kpc
    e_d = d*0.03 #kpc
    
    
    #Correcion por reflejo solar
    c_reflex = gc.reflex_correct(c)
    st_coord_reflex = c_reflex.transform_to(mwsts[st].stream_frame)
    
    pmphi1_reflex = st_coord_reflex.pm_phi1_cosphi2 #mas/yr
    pmphi2_reflex = st_coord_reflex.pm_phi2 #mas/yr
    
    
    #Seleciono estrellas dentro de la mascara
    ra = data['RA_ICRS'][mask] #deg
    e_ra = data['e_RA_ICRS'][mask]/3600 #deg
    dec = data['DE_ICRS'][mask] #deg
    e_dec = data['e_DE_ICRS'][mask]/3600 #deg
    
    pmra = c.pm_ra_cosdec.value #data['pmRA'][mask] #mas/yr
    e_pmra = data['e_pmRA'][mask] #mas/yr
    pmdec = c.pm_dec.value #data['pmDE'][mask] #mas/yr
    e_pmdec = data['e_pmDE'][mask] #mas/yr
    pmra_pmdec_corr = np.zeros(len(e_pmra))
    
    #Create poly
    field = ac.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    
    skypath = np.loadtxt('catalogs/pal5_extended_skypath.icrs.txt')
    skypath_N = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,1]*u.deg, frame='icrs')
    skypath_S = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,2]*u.deg, frame='icrs')

    # Concatenate N track, S-flipped track and add first point at the end to close the polygon (needed for ADQL)
    on_poly = ac.SkyCoord(ra = np.concatenate((skypath_N.ra,skypath_S.ra[::-1],skypath_N.ra[:1])),
                            dec = np.concatenate((skypath_N.dec,skypath_S.dec[::-1],skypath_N.dec[:1])),
                            unit=u.deg, frame='icrs')

    #Starkman2019 track
    # on_poly = mwsts['Pal5-S20'].create_sky_polygon_footprint_from_track(width=1.5*u.deg, phi2_offset=-0.1*u.deg)

    #Mascara del track del stream
    footprint = galstreams.get_mask_in_poly_footprint(on_poly, field, stream_frame=mwsts[st].stream_frame)
    off = ~footprint
    
    # footprint = mwsts[st].get_mask_in_poly_footprint(field)
    # off = ~mwsts[st].get_mask_in_poly_footprint(field)
    
    #Estrellas del fondo
#     ra_out = ra[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     dec_out = dec[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     e_ra_out = e_ra[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     e_dec_out = e_dec[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]

#     pmra_out = pmra[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     pmdec_out = pmdec[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     e_pmra_out = e_pmra[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     e_pmdec_out = e_pmdec[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     pmra_pmdec_corr_out = pmra_pmdec_corr[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
    
#     d_out = d[off & (data['RA_ICRS'][mask] < 250) & (data['DE_ICRS'][mask] > -10.)]
#     e_d_out = d_out*0.08
    
    #Archivo nuevo de las estrellas del fondo
    data1 = pd.read_csv('catalogs/rrls_in_pal5_bkg.m5_ngc5634_removed.csv', index_col=0)
    mask1 = (data1['D_ps1']>0.) & (data1['D_kpc']<35.) & (data1['pmra'] != 0.) & (data1['pmdec'] !=0.)

    pmra_out = np.array(data1[mask1]['pmra'])
    pmdec_out = np.array(data1[mask1]['pmdec'])
    e_pmra_out = np.array(data1[mask1]['pmra_error'])
    e_pmdec_out = np.array(data1[mask1]['pmdec_error'])
    pmra_pmdec_corr_out = np.array(data1[mask1]['pmra_pmdec_corr'])
    d_out = np.array(data1[mask1]['D_ps1'])
    e_d_out = d_out*0.08
    
    
    #Matriz de covarianza de las estrellas intrinsica + observacional <--No tiene en cuenta correlacion entre pm
#     C_int = np.array([[C11, 0, 0], [0, C22, 0], [0, 0, C33]]) #Matriz de covarianza intrinseca (el stream tiene un ancho distinto a 0, los datos se alejan de la media no solo por el error observacional, sino tambien xq es intrinsecamente disperso): Fija para todas las estrellas
    
#     C_pm = [None for n in range(len(e_pmra))]  
#     for i in range(len(e_pmra)):
#         C_pm[i] = np.array([[e_pmra[i]**2,0],[0,e_pmdec[i]**2]])

#     C_pm_radec = np.array(C_pm)
#     C_pm = gc.transform_pm_cov(c, C_pm_radec, mwsts[st].stream_frame) #Transformo matriz cov de pm al frame del stream

#     C_obs = [None for n in range(len(e_pmra))] #Matriz de covarianza observacional en el frame del stream
#     for i in range(len(e_pmra)):
#         C_obs[i] = np.zeros((3,3))
#         C_obs[i][:2,:2] = C_pm[i]
#         C_obs[i][2,2] = e_d[i]**2

#     C_tot = C_int + C_obs
    
    
    #Matriz de covarianza de las estrellas intrinsica + observacional
    C_pm = [None for n in range(len(e_pmra))]  
    for i in range(len(e_pmra)):
        C_pm[i] = np.array([[e_pmra[i]**2, e_pmra[i]*e_pmdec[i]*pmra_pmdec_corr[i]],
                            [e_pmra[i]*e_pmdec[i]*pmra_pmdec_corr[i], e_pmdec[i]**2]])

    C_pm_radec = np.array(C_pm)
    C_pm = gc.transform_pm_cov(c, C_pm_radec, mwsts[st].stream_frame) #Transformo matriz cov de pm al frame del stream

    C_obs = np.zeros((len(e_pmra),3,3)) #Matriz de covarianza observacional en el frame del stream
    C_obs[:,:2,:2] = C_pm
    C_obs[:,2,2] = e_d**2


    C_tot = C_int + C_obs
    
    
    
    #Valores medio y errores del cluster en phi1 y phi2
    ra_mean, dec_mean = 229.022, -0.112
    # mu1_mean, mu2_mean = 3.78307899, 0.71613004 #o 3.758023767746698, 0.7343094450236292 si transformo mu_ra y mu_dec del paper
    mura_mean, mudec_mean = -2.728, -2.687
    e_mura, e_mudec, rho_mu = 0.022, 0.025, -0.39 #Estos son en (alpha, delta), no tendría que transformar a los errores en phi1 y phi2?
    cov_mu = rho_mu*e_mura*e_mudec #rho_xy = sigma_xy/(sigma_x*sigma_y)
    c_mean = ac.ICRS(ra=ra_mean*u.degree, dec=dec_mean*u.degree, distance=d_mean*u.kpc, pm_ra_cosdec=mura_mean*u.mas/u.yr, pm_dec=mudec_mean*u.mas/u.yr, radial_velocity=0*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
    st_coord = c_mean.transform_to(mwsts[st].stream_frame)

    mu1_mean = st_coord.pm_phi1_cosphi2.value #mas/yr
    mu2_mean = st_coord.pm_phi2.value #mas/yr
    mu12 = np.array([mu1_mean, mu2_mean])
    
    C_mean_radec = np.array([[(e_mura*10)**2, cov_mu*100], [cov_mu*100, (e_mudec*10)**2]])
    C_mean12 = gc.transform_pm_cov(c_mean, C_mean_radec, mwsts[st].stream_frame)
    e_mu1 = np.sqrt(C_mean12[0,0])
    e_mu2 = np.sqrt(C_mean12[1,1])
    cov_mu12 = C_mean12[0,1]
    
    
    #Valores medio y errores del cluster en phi1 y phi2
#     f = fits.open('globular_clusters_Vasiliev2019.fit')
#     globs = f[1].data
#     # skip_globs = globs[(globs['RAJ2000'] > ra_lim[0]) & (globs['RAJ2000'] < ra_lim[1]) & 
#     #                    (globs['DEJ2000'] > dec_lim[0]) & (globs['DEJ2000'] < dec_lim[1]) &
#     #                    (globs['Name'] != 'Pal 5')]
#     vasiliev_pal5 = globs[globs['Name'] == Name]

#     vasiliev_pal5_g_2019 = ac.SkyCoord(ra=vasiliev_pal5['RAJ2000'][0]*u.deg,
#                                        dec=vasiliev_pal5['DEJ2000'][0]*u.deg,
#                                        distance=d_mean*u.kpc,
#                                        pm_ra_cosdec=vasiliev_pal5['pmRA'][0]*u.mas/u.yr,
#                                        pm_dec=vasiliev_pal5['pmDE'][0]*u.mas/u.yr,
#                                        radial_velocity=0*u.km/u.s)
#     vasiliev_pal5_c_2019 = vasiliev_pal5_g_2019.transform_to(mwsts[st].stream_frame)
    
#     # skip_globs = skip_globs
#     vasiliev_pal5 = vasiliev_pal5
#     vasiliev_pal5_g = vasiliev_pal5_g_2019
#     vasiliev_pal5_c = vasiliev_pal5_c_2019
    
#     mu1_mean = vasiliev_pal5_c.pm_phi1_cosphi2.value #mas/yr
#     mu2_mean = vasiliev_pal5_c.pm_phi2.value #mas/yr
#     mu12 = np.array([mu1_mean, mu2_mean])
            
#     n = 10
#     C_mean_radec = np.array([[(vasiliev_pal5['e_pmRA'][0]*n)**2, vasiliev_pal5['corr'][0]*vasiliev_pal5['e_pmRA'][0]*n*vasiliev_pal5['e_pmDE'][0]*n], 
#                        [vasiliev_pal5['corr'][0]*vasiliev_pal5['e_pmRA'][0]*n*vasiliev_pal5['e_pmDE'][0]*n, (vasiliev_pal5['e_pmDE'][0]*n)**2]])
#     C_mean12 = gc.transform_pm_cov(vasiliev_pal5_g, C_mean_radec, mwsts[st].stream_frame)

#     e_mu1 = np.sqrt(C_mean12[0,0])
#     e_mu2 = np.sqrt(C_mean12[1,1])
#     cov_mu12 = C_mean12[0,1]
    
    
    
    
    
    
    #Estrellas del track y del fondo
#     d_in = (d>d_inf) & (d<d_sup)
#     inside = (data['Track'][mask]==1)
#     out = (data['Track'][mask]==0)
#     miembro = inside & (data['Memb'][mask]>0.5)
    
#     if printTrack == 'yes':
#         fig=plt.figure(1,figsize=(12,6))
#         fig.subplots_adjust(wspace=0.3,hspace=0.34,top=0.9,bottom=0.17,left=0.11,right=0.97)
#         ax=fig.add_subplot(121)
#         ax.plot(ra[inside],dec[inside],'.',ms=5)
#         ax.plot(ra[out],dec[out],'.',ms=5)
#         ax.plot(ra[miembro],dec[miembro],'*',c='red',ms=10)
#         ax.set_xlabel('$\\alpha$ (°)')
#         ax.set_ylabel('$\delta$ (°)')
#         ax.set_xlim([max(ra), min(ra)])
#         ax.set_ylim([min(dec), max(dec)])

#         ax=fig.add_subplot(122)
#         ax.scatter(pmra[inside & d_in],pmdec[inside & d_in],s=5, label='in')
#         ax.scatter(pmra[out & d_in],pmdec[out & d_in],s=5, label='out')
#         ax.scatter(pmra[miembro & d_in],pmdec[miembro & d_in],s=50, marker='*',color='red', label='memb')
#         ax.set_xlabel('$\mu_\\alpha*$ ("/año)')
#         ax.set_ylabel('$\mu_\delta$ ("/año)')
#         ax.set_title('${} < d < {}$ kpc'.format(d_inf, d_sup))
#         ax.legend(frameon=False, ncol=3, handlelength=0.1)
#         ax.set_xlim([-5,1])
#         ax.set_ylim([-5,1]);


#         fig2=plt.figure(2,figsize=(12,8))
#         fig2.subplots_adjust(wspace=0.3,hspace=0.34,top=0.95,bottom=0.13,left=0.1,right=0.97)
#         ax2=fig2.add_subplot(221)
#         ax2.plot(phi1[inside],phi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],phi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,phi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],phi2[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\phi_2$ (°)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-3,5])

#         ax2=fig2.add_subplot(222)
#         ax2.plot(phi1[inside],d[inside],'.',ms=5)
#         ax2.plot(phi1[out],d[out],'.',ms=2.5)
#         ax2.plot(phi1[miembro],d[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$d$ (kpc)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([13,25])

#         ax2=fig2.add_subplot(223)
#         ax2.plot(phi1[inside],pmphi1[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi1[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi1_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi1[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_1}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([1,6])

#         ax2=fig2.add_subplot(224)
#         ax2.plot(phi1[inside],pmphi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi2[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_2}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-2.5,2.5]);


#         fig3=plt.figure(3,figsize=(8,6))
#         fig3.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
#         ax3=fig3.add_subplot(111)
#         ax3.plot(ra[inside],dec[inside],'.',ms=5)
#         ax3.plot(ra[out],dec[out],'.',ms=2.5)
#         ax3.plot(track.ra, track.dec, 'k.', ms=1.)
#         ax3.plot(mwsts[st].poly_sc.icrs.ra, mwsts[st].poly_sc.icrs.dec, lw=1.,ls='--', color='black')
#         ax3.set_xlabel('$\\alpha$ (°)')
#         ax3.set_ylabel('$\delta$ (°)')
#         ax3.set_xlim([max(data['RA_ICRS']), min(data['RA_ICRS'])])
#         ax3.set_ylim([min(data['DE_ICRS']), max(data['DE_ICRS'])]);


#         fig.savefig('sample.png')
#         fig2.savefig('track_memb.png')
        # fig3.savefig('track.png')

            
    return data, phi1_t, phi2_t, pmphi1_t, pmphi2_t, phi1, phi2, pmphi1, pmphi2, pmra, pmdec, d, pmphi1_reflex, pmphi2_reflex, mu12, C_mean12, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, pmra_pmdec_corr_out, e_d_out, C_pm_radec, C_tot, footprint, mask



def datos_gaia(tabla, st, Name, C_int, d_mean, d_lim, ra_lim, dec_lim):
    """
    Funcion que devuelve la posicion, movimientos propios y distancia de las estrellas y del track en el frame de la corriente, y los movimientos propios en ar y dec y distancia de las estrellas por fuera del track junto a sus errores
    
    Inputs:
    tabla: Nombre de la tabla donde se encuentran los datos crudos
    st: Nombre de la corriente estelar
    printTrack: Imprime (yes/no) el track en ar y dec, pm_ra y pm_dec, phi1 y phi2, phi1 y d, phi1 y pmphi1, phi1 y pmphi2
    C11, C22, C33: Valores de la matriz de covarianza intriseca del stream en movimientos propios y distancia
    ra_mean, dec_mean: Ascención recta y declinacion medios del stream
    mura_mean, mura_mean: Movimientos propios medios en ascención recta y declinacion 
    e_mura, e_mura, cov_mu: Errores de los movimientos propios medios en ascencion recta y declinacion, y el factor de correlacion entre ambos
    d_inf, d_sup: Distancia minima y maxima de la corriente
    cut_lim: Limites inferior y superior en la mascara de corte en distancia

    
    Outputs:
    data: Tabla con los datos originales
    phi1, phi2: Posicion de las estrellas en el frame de la corriente
    pmphi1, pmphi2: Moviemientos propios de las estrellas en el frame de la corriente
    pmphi1_reflex, pmphi2_reflex: Moviemientos propios de las estrellas en el frame de la corriente corregidas por el reflejo solar
    pmra, pmdec, d: Movimientos propios en ascención recta y declinacion y distancias de las estrellas
    phi1_t, phi2_t, pmphi1_t, pmphi2_t: Posicion y movimientos propios del track en el frame de la corriente
    pmra_out, pmdec_out, d_out: movimientos propios y distancia de las estrellas fuera del track en ar y dec
    e_pmra_out, e_pmdec_out, e_d_out: Errores en los movimientos propios y distancia de las estrellas fuera del track en ar y dec
    C_tot: Matriz de covarianza intrinseca (fija) + observacional en el frame del stream
    footprint: Mascara con las estrellas espacialmente dentro del track
    d_lim: Corte en distancia de las estrellas que no voy a tomar en cuenta
    """

    print('\nCargando datos \n')

    ##Cargo datos
    
    data = pd.read_csv('catalogs/g_all.csv')

    print('Cargando track y transformando coordenadas \n')
    
    ##Cargo track y transformo coordenadas
    mwsts = galstreams.MWStreams(verbose=False, implement_Off=True)
    track = mwsts[st].track
    #track = gc.reflex_correct(track) #Por ahora no uso la correcion por reflejo del sol
    st_track = track.transform_to(mwsts[st].stream_frame)
    
    phi1_t = st_track.phi1
    phi2_t = st_track.phi2
    pmphi1_t = st_track.pm_phi1_cosphi2
    pmphi2_t = st_track.pm_phi2
    
    
    #mascara en datos
    mask = (data['D_ps1'] > d_lim[0]) & (data['D_kpc'] < d_lim[1]) & (data['pmra'] != 0.) & (data['pmdec'] !=0.)# & (data['ra'] < 250) & (data['dec'] > -10.)

    
    c = ac.ICRS(ra=data['ra'][mask]*u.degree, dec=data['dec'][mask]*u.degree, distance=data['D_ps1'][mask]*u.kpc, pm_ra_cosdec=data['pmra'][mask]*u.mas/u.yr, pm_dec=data['pmdec'][mask]*u.mas/u.yr, radial_velocity=np.zeros(len(data['pmra'][mask]))*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
    st_coord = c.transform_to(mwsts[st].stream_frame)
    
    
    phi1 = st_coord.phi1 #deg
    phi2 = st_coord.phi2 #deg
    pmphi1 = st_coord.pm_phi1_cosphi2 #mas/yr
    pmphi2 = st_coord.pm_phi2 #mas/yr
    d = np.array(data['D_ps1'][mask]) #kpc
    e_d = d*0.03 #kpc
    
    
    #Correcion por reflejo del sol
    c_reflex = gc.reflex_correct(c)
    st_coord_reflex = c_reflex.transform_to(mwsts[st].stream_frame)
    pmphi1_reflex = st_coord_reflex.pm_phi1_cosphi2 #mas/yr
    pmphi2_reflex = st_coord_reflex.pm_phi2 #mas/yr
    
    
    ra = np.array(data['ra'][mask]) #deg
    e_ra = np.array(data['ra_error'][mask])/3600 #deg
    dec = np.array(data['dec'][mask]) #deg
    e_dec = np.array(data['dec_error'][mask])/3600 #deg
    
    pmra = c.pm_ra_cosdec.value #data['pmRA'][mask] #mas/yr
    e_pmra = np.array(data['pmra_error'][mask]) #mas/yr
    pmdec = c.pm_dec.value #data['pmDE'][mask] #mas/yr
    e_pmdec = np.array(data['pmdec_error'][mask]) #mas/yr
    pmra_pmdec_corr = np.array(data['pmra_pmdec_corr'][mask]) 
    
    #Create poly del track
    field = ac.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    
    skypath = np.loadtxt('catalogs/pal5_extended_skypath.icrs.txt')
    skypath_N = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,1]*u.deg, frame='icrs')
    skypath_S = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,2]*u.deg, frame='icrs')

    # Concatenate N track, S-flipped track and add first point at the end to close the polygon (needed for ADQL)
    on_poly = ac.SkyCoord(ra = np.concatenate((skypath_N.ra,skypath_S.ra[::-1],skypath_N.ra[:1])),
                            dec = np.concatenate((skypath_N.dec,skypath_S.dec[::-1],skypath_N.dec[:1])),
                            unit=u.deg, frame='icrs')
    
    #Starkman2019 track
    # on_poly = mwsts['Pal5-S20'].create_sky_polygon_footprint_from_track(width=1.5*u.deg, phi2_offset=-0.1*u.deg)

    #Select the field points inside the polygon footprint
    footprint = galstreams.get_mask_in_poly_footprint(on_poly, field, stream_frame=mwsts[st].stream_frame)
    off = ~footprint
    

    #Estrellas del fondo
#     ra_out = ra[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
#     dec_out = dec[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
#     e_ra_out = e_ra[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
#     e_dec_out = e_dec[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]

#     pmra_out = pmra[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
#     pmdec_out = pmdec[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
#     e_pmra_out = e_pmra[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
#     e_pmdec_out = e_pmdec[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
    
#     d_out = d[off]# & (data['ra'][mask] < 250) & (data['dec'][mask] > -10.)]
#     e_d_out = d_out*0.08
    
    
    data1 = pd.read_csv('catalogs/rrls_in_pal5_bkg.m5_ngc5634_removed.csv', index_col=0)
    mask1 = (data1['D_ps1']>0.) & (data1['D_kpc']<35.) & (data1['pmra'] != 0.) & (data1['pmdec'] !=0.)

    pmra_out = np.array(data1[mask1]['pmra'])
    pmdec_out = np.array(data1[mask1]['pmdec'])
    e_pmra_out = np.array(data1[mask1]['pmra_error'])
    e_pmdec_out = np.array(data1[mask1]['pmdec_error'])
    pmra_pmdec_corr_out = np.array(data1[mask1]['pmra_pmdec_corr'])
    d_out = np.array(data1[mask1]['D_ps1'])
    e_d_out = d_out*0.08

    
    
    #Matriz de covarianza de las estrellas intrinsica + observacional
    C_pm = [None for n in range(len(e_pmra))]  
    for i in range(len(e_pmra)):
        C_pm[i] = np.array([[e_pmra[i]**2, e_pmra[i]*e_pmdec[i]*pmra_pmdec_corr[i]],
                            [e_pmra[i]*e_pmdec[i]*pmra_pmdec_corr[i], e_pmdec[i]**2]])

    C_pm_radec = np.array(C_pm)
    C_pm = gc.transform_pm_cov(c, C_pm_radec, mwsts[st].stream_frame) #Transformo matriz cov de pm al frame del stream

    C_obs = np.zeros((len(e_pmra),3,3)) #Matriz de covarianza observacional en el frame del stream
    C_obs[:,:2,:2] = C_pm
    C_obs[:,2,2] = e_d**2


    C_tot = C_int + C_obs
    
    
    #Valores medio y errores del cluster en phi1 y phi2
#     ra_mean, dec_mean = 229.022, -0.112
#     # mu1_mean, mu2_mean = 3.78307899, 0.71613004 #o 3.758023767746698, 0.7343094450236292 si transformo mu_ra y mu_dec del paper
#     mura_mean, mudec_mean = -2.728, -2.687
#     e_mura, e_mudec, rho_mu = 0.022, 0.025, -0.39 #Estos son en (alpha, delta), no tendría que transformar a los errores en phi1 y phi2?
#     cov_mu = rho_mu*e_mura*e_mudec #rho_xy = sigma_xy/(sigma_x*sigma_y)
#     c_mean = ac.ICRS(ra=ra_mean*u.degree, dec=dec_mean*u.degree, distance=d_mean*u.kpc, pm_ra_cosdec=mura_mean*u.mas/u.yr, pm_dec=mudec_mean*u.mas/u.yr, radial_velocity=0*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
#     st_coord = c_mean.transform_to(mwsts[st].stream_frame)

#     mu1_mean = st_coord.pm_phi1_cosphi2.value #mas/yr
#     mu2_mean = st_coord.pm_phi2.value #mas/yr
#     mu12 = np.array([mu1_mean, mu2_mean])
    
#     C_mean_radec = np.array([[(e_mura*10)**2, cov_mu*100], [cov_mu*100, (e_mudec*10)**2]])
#     C_mean12 = gc.transform_pm_cov(c_mean, C_mean_radec, mwsts[st].stream_frame)
#     e_mu1 = np.sqrt(C_mean12[0,0])
#     e_mu2 = np.sqrt(C_mean12[1,1])
#     cov_mu12 = C_mean12[0,1]
    
    
    #Valores medio y errores del cluster en phi1 y phi2
    f = fits.open('catalogs/globular_clusters_Vasiliev&Baumgardt2021.fit')
    globs = f[1].data
    # skip_globs = globs[(globs['RAJ2000'] > ra_lim[0]) & (globs['RAJ2000'] < ra_lim[1]) & 
    #                    (globs['DEJ2000'] > dec_lim[0]) & (globs['DEJ2000'] < dec_lim[1]) &
    #                    (globs['Name'] != 'Pal 5')]
    vasiliev_pal5 = globs[globs['Name'] == Name]

    vasiliev_pal5_g_2019 = ac.SkyCoord(ra=vasiliev_pal5['RAJ2000'][0]*u.deg,
                                       dec=vasiliev_pal5['DEJ2000'][0]*u.deg,
                                       distance=d_mean*u.kpc,
                                       pm_ra_cosdec=vasiliev_pal5['pmRA'][0]*u.mas/u.yr,
                                       pm_dec=vasiliev_pal5['pmDE'][0]*u.mas/u.yr,
                                       radial_velocity=0*u.km/u.s)
    vasiliev_pal5_c_2019 = vasiliev_pal5_g_2019.transform_to(mwsts[st].stream_frame)
    
    # skip_globs = skip_globs
    vasiliev_pal5 = vasiliev_pal5
    vasiliev_pal5_g = vasiliev_pal5_g_2019
    vasiliev_pal5_c = vasiliev_pal5_c_2019
    
    mu1_mean = vasiliev_pal5_c.pm_phi1_cosphi2.value #mas/yr
    mu2_mean = vasiliev_pal5_c.pm_phi2.value #mas/yr
    mu12 = np.array([mu1_mean, mu2_mean])
            
    n = 10
    C_mean_radec = np.array([[(vasiliev_pal5['e_pmRA'][0]*n)**2, vasiliev_pal5['corr'][0]*vasiliev_pal5['e_pmRA'][0]*n*vasiliev_pal5['e_pmDE'][0]*n], 
                       [vasiliev_pal5['corr'][0]*vasiliev_pal5['e_pmRA'][0]*n*vasiliev_pal5['e_pmDE'][0]*n, (vasiliev_pal5['e_pmDE'][0]*n)**2]])
    C_mean12 = gc.transform_pm_cov(vasiliev_pal5_g, C_mean_radec, mwsts[st].stream_frame)

    e_mu1 = np.sqrt(C_mean12[0,0])
    e_mu2 = np.sqrt(C_mean12[1,1])
    cov_mu12 = C_mean12[0,1]



    
    
    #Estrellas del track y del fondo
#     d_in = (d>d_inf) & (d<d_sup)
#     inside = (data['Track'][mask]==1)
#     out = (data['Track'][mask]==0)
#     miembro = inside & (data['Memb'][mask]>0.5)
    
#     if printTrack == 'yes':
#         fig=plt.figure(1,figsize=(12,6))
#         fig.subplots_adjust(wspace=0.3,hspace=0.34,top=0.9,bottom=0.17,left=0.11,right=0.97)
#         ax=fig.add_subplot(121)
#         ax.plot(ra[inside],dec[inside],'.',ms=5)
#         ax.plot(ra[out],dec[out],'.',ms=5)
#         ax.plot(ra[miembro],dec[miembro],'*',c='red',ms=10)
#         ax.set_xlabel('$\\alpha$ (°)')
#         ax.set_ylabel('$\delta$ (°)')
#         ax.set_xlim([max(ra), min(ra)])
#         ax.set_ylim([min(dec), max(dec)])

#         ax=fig.add_subplot(122)
#         ax.scatter(pmra[inside & d_in],pmdec[inside & d_in],s=5, label='in')
#         ax.scatter(pmra[out & d_in],pmdec[out & d_in],s=5, label='out')
#         ax.scatter(pmra[miembro & d_in],pmdec[miembro & d_in],s=50, marker='*',color='red', label='memb')
#         ax.set_xlabel('$\mu_\\alpha*$ ("/año)')
#         ax.set_ylabel('$\mu_\delta$ ("/año)')
#         ax.set_title('${} < d < {}$ kpc'.format(d_inf, d_sup))
#         ax.legend(frameon=False, ncol=3, handlelength=0.1)
#         ax.set_xlim([-5,1])
#         ax.set_ylim([-5,1]);


#         fig2=plt.figure(2,figsize=(12,8))
#         fig2.subplots_adjust(wspace=0.3,hspace=0.34,top=0.95,bottom=0.13,left=0.1,right=0.97)
#         ax2=fig2.add_subplot(221)
#         ax2.plot(phi1[inside],phi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],phi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,phi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],phi2[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\phi_2$ (°)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-3,5])

#         ax2=fig2.add_subplot(222)
#         ax2.plot(phi1[inside],d[inside],'.',ms=5)
#         ax2.plot(phi1[out],d[out],'.',ms=2.5)
#         ax2.plot(phi1[miembro],d[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$d$ (kpc)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([13,25])

#         ax2=fig2.add_subplot(223)
#         ax2.plot(phi1[inside],pmphi1[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi1[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi1_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi1[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_1}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([1,6])

#         ax2=fig2.add_subplot(224)
#         ax2.plot(phi1[inside],pmphi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi2[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_2}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-2.5,2.5]);


#         fig3=plt.figure(3,figsize=(8,6))
#         fig3.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
#         ax3=fig3.add_subplot(111)
#         ax3.plot(ra[inside],dec[inside],'.',ms=5)
#         ax3.plot(ra[out],dec[out],'.',ms=2.5)
#         ax3.plot(track.ra, track.dec, 'k.', ms=1.)
#         ax3.plot(mwsts[st].poly_sc.icrs.ra, mwsts[st].poly_sc.icrs.dec, lw=1.,ls='--', color='black')
#         ax3.set_xlabel('$\\alpha$ (°)')
#         ax3.set_ylabel('$\delta$ (°)')
#         ax3.set_xlim([max(data['RA_ICRS']), min(data['RA_ICRS'])])
#         ax3.set_ylim([min(data['DE_ICRS']), max(data['DE_ICRS'])]);


#         fig.savefig('sample.png')
#         fig2.savefig('track_memb.png')
        # fig3.savefig('track.png')

    return data, phi1_t, phi2_t, pmphi1_t, pmphi2_t, phi1, phi2, pmphi1, pmphi2, pmra, pmdec, d, pmphi1_reflex, pmphi2_reflex, mu12, C_mean12, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, pmra_pmdec_corr_out, e_d_out, C_pm_radec, C_tot, footprint, mask



def datos_gaiaDR3(st, Name, Name_d, width, C_int, d_lim, ra_lim, dec_lim):
    """
    Funcion que devuelve la posicion, movimientos propios y distancia de las estrellas y del track en el frame de la corriente, y los movimientos propios en ar y dec y distancia de las estrellas por fuera del track junto a sus errores
    
    Inputs:
    tabla: Nombre de la tabla donde se encuentran los datos crudos
    st: Nombre de la corriente estelar
    printTrack: Imprime (yes/no) el track en ar y dec, pm_ra y pm_dec, phi1 y phi2, phi1 y d, phi1 y pmphi1, phi1 y pmphi2
    C11, C22, C33: Valores de la matriz de covarianza intriseca del stream en movimientos propios y distancia
    ra_mean, dec_mean: Ascención recta y declinacion medios del stream
    mura_mean, mura_mean: Movimientos propios medios en ascención recta y declinacion 
    e_mura, e_mura, cov_mu: Errores de los movimientos propios medios en ascencion recta y declinacion, y el factor de correlacion entre ambos
    d_inf, d_sup: Distancia minima y maxima de la corriente
    cut_lim: Limites inferior y superior en la mascara de corte en distancia

    
    Outputs:
    data: Tabla con los datos originales
    phi1, phi2: Posicion de las estrellas en el frame de la corriente
    pmphi1, pmphi2: Moviemientos propios de las estrellas en el frame de la corriente
    pmphi1_reflex, pmphi2_reflex: Moviemientos propios de las estrellas en el frame de la corriente corregidas por el reflejo solar
    pmra, pmdec, d: Movimientos propios en ascención recta y declinacion y distancias de las estrellas
    phi1_t, phi2_t, pmphi1_t, pmphi2_t: Posicion y movimientos propios del track en el frame de la corriente
    pmra_out, pmdec_out, d_out: movimientos propios y distancia de las estrellas fuera del track en ar y dec
    e_pmra_out, e_pmdec_out, e_d_out: Errores en los movimientos propios y distancia de las estrellas fuera del track en ar y dec
    C_tot: Matriz de covarianza intrinseca (fija) + observacional en el frame del stream
    footprint: Mascara con las estrellas espacialmente dentro del track
    d_lim: Corte en distancia de las estrellas que no voy a tomar en cuenta
    """

    print('\nCargando datos \n')
    
    ##Cargo track y transformo coordenadas
    mwsts = galstreams.MWStreams(verbose=False, implement_Off=True)
    track = mwsts[st].track
    #track = gc.reflex_correct(track) #Por ahora no uso la correcion por reflejo del sol
    st_track = track.transform_to(mwsts[st].stream_frame)
    
    phi1_t = st_track.phi1
    phi2_t = st_track.phi2
    pmphi1_t = st_track.pm_phi1_cosphi2
    pmphi2_t = st_track.pm_phi2
    d_t = st_track.distance
    
    ##Valores medio y errores del cluster en phi1 y phi2
    globs_d = pd.read_csv('../catalogs/globular_clusters_Baumgardt&Vasiliev2021.txt', header=0, sep='\s+')
    baumgardt_st = globs_d[globs_d.Cluster == Name_d]
    d_mean = float(baumgardt_st.Rsun)
    e_d_mean = float(baumgardt_st.e_Rsun)
    
    
    f = fits.open('../catalogs/globular_clusters_Vasiliev&Baumgardt2021.fit')
    globs = f[1].data
    
    vasiliev_st = globs[globs['Name'] == Name]

    vasiliev_st_g = ac.SkyCoord(ra=vasiliev_st['RAJ2000'][0]*u.deg,
                                  dec=vasiliev_st['DEJ2000'][0]*u.deg,
                                  distance=d_mean*u.kpc,
                                  pm_ra_cosdec=vasiliev_st['pmRA'][0]*u.mas/u.yr,
                                  pm_dec=vasiliev_st['pmDE'][0]*u.mas/u.yr,
                                  radial_velocity=0*u.km/u.s)
    
    vasiliev_st_c = vasiliev_st_g.transform_to(mwsts[st].stream_frame)

    
    mu1_mean = vasiliev_st_c.pm_phi1_cosphi2.value #mas/yr
    mu2_mean = vasiliev_st_c.pm_phi2.value #mas/yr
    mu12_mean = np.array([mu1_mean, mu2_mean])
    
    
            
    n = 10
    C_mean_radec = np.array([[(vasiliev_st['e_pmRA'][0]*n)**2, vasiliev_st['corr'][0]*vasiliev_st['e_pmRA'][0]*n*vasiliev_st['e_pmDE'][0]*n], 
                       [vasiliev_st['corr'][0]*vasiliev_st['e_pmRA'][0]*n*vasiliev_st['e_pmDE'][0]*n, (vasiliev_st['e_pmDE'][0]*n)**2]])
    C_mean12 = gc.transform_pm_cov(vasiliev_st_g, C_mean_radec, mwsts[st].stream_frame)

    e_mu1 = np.sqrt(C_mean12[0,0])
    e_mu2 = np.sqrt(C_mean12[1,1])
    cov_mu12 = C_mean12[0,1]

    
    #Cargo datos
    skip_globs = globs[(globs['RAJ2000'] > ra_lim[0]) & (globs['RAJ2000'] < ra_lim[1]) & 
                       (globs['DEJ2000'] > dec_lim[0]) & (globs['DEJ2000'] < dec_lim[1]) &
                       (globs['Name'] != Name)]
    
    
    def skip_mask(ra, dec):
        c1 = ac.SkyCoord(ra, dec)
        c2 = ac.SkyCoord(skip_globs['RAJ2000']*u.deg, skip_globs['DEJ2000']*u.deg)

        mask = np.ones(len(ra), dtype=bool)
        for c in c2:
            mask &= c1.separation(c) > 0.5*u.deg
    
        return mask


    # data = pd.read_csv('g_all.csv')
    _tbl = Table.read('../catalogs/gaiaDR3/masterv4.rrls.gapzo.gdr3-full.csv.gz', format='ascii.csv')
    _tbl_dist = Table.read('../catalogs/gaiaDR3/masterv4.rrls.gapzo.dist.short.csv.gz', format='ascii.csv') #OverflowError warning

    _tbl['Dist'] = _tbl_dist['Dist']
    _tbl['Dist_err'] = _tbl_dist['Dist_err']

    g_all = GaiaData(_tbl)
    
    mask = (g_all.ra > ra_lim[0]*u.deg) & (g_all.ra < ra_lim[1]*u.deg) & (g_all.dec > dec_lim[0]*u.deg) & (g_all.dec < dec_lim[1]*u.deg) & skip_mask(g_all.ra, g_all.dec) & (g_all.pmra.value!=1e20) & (g_all.pmdec.value != 1e20) & (g_all.pmra.value != 0.) & (g_all.pmdec.value != 0.) & (g_all.Dist > d_lim[0]) & (g_all.Dist < d_lim[1]) & (g_all.dec > (((60+70)*u.deg)/((305-190)*u.deg))*(g_all.ra-305*u.deg)+60*u.deg) & (g_all.b > 10*u.deg)
    
    g_all = g_all[mask]
    
    c_all = g_all.get_skycoord(distance=g_all.Dist*u.kpc, radial_velocity = np.zeros(len(g_all.Dist))*u.km/u.s)
    c_all_st = c_all.transform_to(mwsts[st].stream_frame)
    
    
    
    
    #Le agrego gigante de Odenkirchen a ver si que pasa
#     t1 = Table.read('catalogs/Odenkirchen2002_gaia.csv')
#     t2 = Table.read('catalogs/Odenkirchen2009_gaia.csv')
#     Oden = astropy.table.vstack((t1, t2))
#     Oden.rename_column('ra','ra_O')
#     Oden.rename_column('dec','dec_O')
#     Oden.rename_column('RAdeg','ra')
#     Oden.rename_column('DEdeg','dec')
#     Oden.rename_column('e_RAdeg','ra_error')
#     Oden.rename_column('e_DEdeg','dec_error')
#     Oden.rename_column('pmRA', 'pmra')
#     Oden.rename_column('pmDE', 'pmdec')
#     Oden.rename_column('e_pmRA', 'pmra_error')
#     Oden.rename_column('e_pmDE', 'pmdec_error')
#     Oden.rename_column('pmRApmDEcor', 'pmra_pmdec_corr')

#     g_oden = GaiaData(Oden)
#     g_oden = g_oden[(g_oden.vr_a > -80) & (g_oden.vr_a < -20) & (g_oden.pmra > -5*u.mas/u.yr)]
#     g_oden = g_oden[(g_oden.vr_a > -66) & (g_oden.vr_a < -50) & (g_oden.ra < 232*u.deg)]
#     g_oden.data['Dist'] = 21. # HACK
#     g_oden.data['Dist_err'] = 10. # HACK
    
#     g_merged = GaiaData(astropy.table.vstack((g_all.data, g_oden.data)))
#     merged_c = g_merged.get_skycoord(distance=g_merged.Dist*u.kpc, radial_velocity = np.zeros(len(g_merged.Dist))*u.km/u.s)
#     merged_c_pal5 = merged_c.transform_to(mwsts[st].stream_frame)
    
#     g_all = g_merged
#     c_all = merged_c
#     c_all_st = merged_c_pal5
    
    
    
    
    
    phi1 = c_all_st.phi1 #deg
    phi2 = c_all_st.phi2 #deg
    pmphi1 = c_all_st.pm_phi1_cosphi2 #mas/yr
    pmphi2 = c_all_st.pm_phi2 #mas/yr
    d = g_all.Dist #kpc
    e_d = g_all.Dist_err #kpc

    
    ra = g_all.ra.value #deg
    e_ra = g_all.ra_error.value/3600 #deg
    dec = g_all.dec.value #deg
    e_dec = g_all.dec_error.value/3600 #deg
    
    pmra = c_all.pm_ra_cosdec.value #data['pmRA'][mask] #mas/yr
    e_pmra = g_all.pmra_error.value #mas/yr
    pmdec = c_all.pm_dec.value #data['pmDE'][mask] #mas/yr
    e_pmdec = g_all.pmdec_error.value #mas/yr
    pmra_pmdec_corr = g_all.pmra_pmdec_corr
    
    #Correccion por reflejo solar
    c_reflex = gc.reflex_correct(c_all)
    st_coord_reflex = c_reflex.transform_to(mwsts[st].stream_frame)
    pmphi1_reflex = st_coord_reflex.pm_phi1_cosphi2 #mas/yr
    pmphi2_reflex = st_coord_reflex.pm_phi2 #mas/yr
    
    
    #Matriz de covarianza de las estrellas intrinsica + observacional 
    C_pm_radec = g_all.get_cov()[:, 3:5, 3:5]
    C_pm = gc.transform_pm_cov(c_all, C_pm_radec, mwsts[st].stream_frame)
    C_obs = np.zeros((len(e_pmra),3,3)) #Matriz de covarianza observacional en el frame del stream
    C_obs[:,:2,:2] = C_pm
    C_obs[:,2,2] = e_d**2


    C_tot = C_int + C_obs
    
    
    #Create poly del track    
#     skypath = np.loadtxt('catalogs/pal5_extended_skypath.icrs.txt')
#     skypath_N = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,1]*u.deg, frame='icrs')
#     skypath_S = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,2]*u.deg, frame='icrs')

#     # Concatenate N track, S-flipped track and add first point at the end to close the polygon (needed for ADQL)
#     on_poly = ac.SkyCoord(ra = np.concatenate((skypath_N.ra,skypath_S.ra[::-1],skypath_N.ra[:1])),
#                             dec = np.concatenate((skypath_N.dec,skypath_S.dec[::-1],skypath_N.dec[:1])),
#                             unit=u.deg, frame='icrs')

    
    #Nate track. Starkman+2019
    # on_poly = mwsts['Pal5-S20'].create_sky_polygon_footprint_from_track(width=1.5*u.deg, phi2_offset=-0.1*u.deg)
    
    #st track
    on_poly = mwsts[st].create_sky_polygon_footprint_from_track(width=width*u.deg, phi2_offset=0.*u.deg)
    
    
    #Select the field points inside the polygon footprint
    field = ac.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    footprint = galstreams.get_mask_in_poly_footprint(on_poly, field, stream_frame=mwsts[st].stream_frame)
    off = ~footprint
    

    #Estrellas del fondo
#     ra_out = ra[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     dec_out = dec[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     e_ra_out = e_ra[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     e_dec_out = e_dec[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]

#     pmra_out = pmra[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     pmdec_out = pmdec[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     e_pmra_out = e_pmra[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     e_pmdec_out = e_pmdec[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     pmra_pmdec_corr_out = pmra_pmdec_corr[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]

#     d_out = d[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]
#     e_d_out = e_d[off]# & (g_all.ra.value< 250) & (g_all.dec.value> -10.)]

    
    
    
#     data1 = pd.read_csv('rrls_in_pal5_bkg.m5_ngc5634_removed.csv', index_col=0)
#     mask1 = (data1['D_ps1']>0.) & (data1['D_kpc']<35.) & (data1['pmra'] != 0.) & (data1['pmdec'] !=0.)

#     pmra_out = np.array(data1[mask1]['pmra'])
#     pmdec_out = np.array(data1[mask1]['pmdec'])
#     e_pmra_out = np.array(data1[mask1]['pmra_error'])
#     e_pmdec_out = np.array(data1[mask1]['pmdec_error'])
#     pmra_pmdec_corr_out = np.array(data1[mask1]['pmra_pmdec_corr'])
#     d_out = np.array(data1[mask1]['D_ps1'])
#     e_d_out = d_out*0.08




    #Estrellas del track y del fondo
    # d_inf, d_sup = 18,25 #Pal5
    # d_inf, d_sup = 3,4 #Gjoll
    d_inf, d_sup = 0,10 #Fjorm
    d_in = (d>d_inf) & (d<d_sup)
    inside = footprint
    out = off
    printTrack = 'no'
    
    if printTrack == 'yes':
        fig=plt.figure(1,figsize=(12,6))
        fig.subplots_adjust(wspace=0.3,hspace=0.34,top=0.9,bottom=0.17,left=0.11,right=0.97)
        ax=fig.add_subplot(121)
        ax.plot(ra[inside],dec[inside],'.',c='black',ms=5)
        ax.plot(ra[out],dec[out],'.',c='gray',ms=1.5)
        ax.set_xlabel('$\\alpha$ (°)')
        ax.set_ylabel('$\delta$ (°)')
        ax.set_xlim([max(ra), min(ra)])
        ax.set_ylim([min(dec), max(dec)])

        ax=fig.add_subplot(122)
        ax.scatter(pmra[inside & d_in],pmdec[inside & d_in],s=5, color='black',label='in')
        ax.scatter(pmra[out & d_in],pmdec[out & d_in],s=1.5, color='gray', label='out')
        ax.plot(track.pm_ra_cosdec, track.pm_dec,'-k', lw=2.5)
        
        ax.set_xlabel('$\mu_\\alpha*$ ("/año)')
        ax.set_ylabel('$\mu_\delta$ ("/año)')
        ax.set_title('${} < d < {}$ kpc'.format(d_inf, d_sup))
        ax.legend(frameon=False, ncol=3, handlelength=0.1)
        ax.set_xlim([-5,1])
        ax.set_ylim([-5,1]);


        fig2=plt.figure(2,figsize=(12,8))
        fig2.subplots_adjust(wspace=0.3,hspace=0.34,top=0.95,bottom=0.13,left=0.1,right=0.97)
        ax2=fig2.add_subplot(221)
        ax2.plot(phi1[inside],phi2[inside],'.',c='black',ms=5)
        ax2.plot(phi1[out],phi2[out],'.',c='gray',ms=1.5)
        ax2.plot(phi1_t,phi2_t,'k-',lw=2.5)
        # ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$\phi_2$ (°)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([-3,5])

        ax2=fig2.add_subplot(222)
        ax2.plot(phi1[inside],d[inside],'.',color='black',ms=5)
        ax2.plot(phi1[out],d[out],'.',c='gray',ms=1.5)
        ax2.plot(phi1_t,d_t,'k-',lw=2.5)
        # ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$d$ (kpc)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([13,25])

        ax2=fig2.add_subplot(223)
        ax2.plot(phi1[inside],pmphi1[inside],'.',c='black',ms=5)
        ax2.plot(phi1[out],pmphi1[out],'.',c='gray',ms=1.5)
        ax2.plot(phi1_t,pmphi1_t,'k-',lw=2.5)
        ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$\mu_{\phi_1}$ (mas/yr)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([1,6])

        ax2=fig2.add_subplot(224)
        ax2.plot(phi1[inside],pmphi2[inside],'.',c='black',ms=5)
        ax2.plot(phi1[out],pmphi2[out],'.',c='gray',ms=2.5)
        ax2.plot(phi1_t,pmphi2_t,'k-',lw=2.5)
        ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$\mu_{\phi_2}$ (mas/yr)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([-2.5,2.5]);


        # fig3=plt.figure(3,figsize=(8,6))
        # fig3.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
        # ax3=fig3.add_subplot(111)
        # ax3.plot(ra[inside],dec[inside],'.',c='black',ms=5)
        # ax3.plot(ra[out],dec[out],'.',c='gray',ms=1.5)
        # ax3.plot(track.ra, track.dec, 'k-', lw=2.5)
        # ax3.plot(on_poly.icrs.ra, on_poly.icrs.ra, ls='--', lw=1., color='black')
        # ax3.set_xlabel('$\\alpha$ (°)')
        # ax3.set_ylabel('$\delta$ (°)')
        # ax3.set_xlim([max(ra), min(ra)])
        # ax3.set_ylim([min(dec), max(dec)]);


    return g_all, phi1_t, phi2_t, pmphi1_t, pmphi2_t, d_t, phi1, phi2, pmphi1, pmphi2, pmra, pmdec, d, C_pm_radec, e_d, pmphi1_reflex, pmphi2_reflex, mu12_mean, C_mean12, d_mean, e_d_mean, C_tot, footprint, mask











# def datos_sinsgr(tabla, st, printTrack, C11, C22, C33, d_mean, ra_mean, dec_mean, mura_mean, mudec_mean, e_mura, e_mudec, cov_mu, d_inf, d_sup):
#     """
#     Funcion que devuelve la posicion, movimientos propios y distancia de las estrellas y del track en el frame de la corriente, y los movimientos propios en ar y dec y distancia de las estrellas por fuera del track junto a sus errores
    
#     Inputs:
#     tabla: Nombre de la tabla donde se encuentran los datos crudos
#     st: Nombre de la corriente estelar
#     printTrack: Imprime (yes/no) el track en ar y dec, pm_ra y pm_dec, phi1 y phi2, phi1 y d, phi1 y pmphi1, phi1 y pmphi2
#     C11, C22, C33: Valores de la matriz de covarianza intriseca del stream en movimientos propios y distancia
#     ra_mean, dec_mean: Ascención recta y declinacion medios del stream
#     mura_mean, mura_mean: Movimientos propios medios en ascención recta y declinacion 
#     e_mura, e_mura, cov_mu: Errores de los movimientos propios medios en ascencion recta y declinacion, y el factor de correlacion entre ambos
#     d_inf: Distancia minima de la corriente
#     d_sup: Distancia maxima de la corriente
    
#     Outputs:
#     data: Tabla con los datos originales
#     phi1, phi2: Posicion de las estrellas en el frame de la corriente
#     pmphi1, pmphi2: Moviemientos propios de las estrellas en el frame de la corriente
#     pmphi1_reflex, pmphi2_reflex: Moviemientos propios de las estrellas en el frame de la corriente corregidas por el reflejo solar
#     pmra, pmdec, d: Movimientos propios en ascención recta y declinacion y distancias de las estrellas
#     phi1_t, phi2_t, pmphi1_t, pmphi2_t: Posicion y movimientos propios del track en el frame de la corriente
#     pmra_out, pmdec_out, d_out: movimientos propios y distancia de las estrellas fuera del track en ar y dec
#     e_pmra_out, e_pmdec_out, e_d_out: Errores en los movimientos propios y distancia de las estrellas fuera del track en ar y dec
#     C_tot: Matriz de covarianza intrinseca (fija) + observacional en el frame del stream
#     footprint: Mascara con las estrellas espacialmente dentro del track
#     """

#     print('\nCargando datos \n')

#     ##Cargo datos
#     f = fits.open(tabla)
#     data = f[1].data
#     # data.columns
#     # Table(data)

#     print('Cargando track y transformando coordenadas \n')
    
#     ##Cargo track y transformo coordenadas
#     mwsts = galstreams.MWStreams(verbose=False, implement_Off=False)
#     track = mwsts[st].track
#     #track = gc.reflex_correct(track) #Por ahora no uso la correcion por reflejo del sol
#     st_track = track.transform_to(mwsts[st].stream_frame)
    
#     phi1_t = st_track.phi1
#     phi2_t = st_track.phi2
#     pmphi1_t = st_track.pm_phi1_cosphi2
#     pmphi2_t = st_track.pm_phi2
    
    
#     sgr = data['Dist'] > 40 #Creo máscara para sacar a la corriente de Sagitario de la ecuacion
    
#     _ = ac.galactocentric_frame_defaults.set('v4.0') #set the default Astropy Galactocentric frame parameters to the values adopted in Astropy v4.0
    
#     c = ac.ICRS(ra=data['RA_ICRS']*u.degree, dec=data['DE_ICRS']*u.degree, distance=data['Dist']*u.kpc, pm_ra_cosdec=data['pmRA']*u.mas/u.yr, pm_dec=data['pmDE']*u.mas/u.yr, radial_velocity=np.zeros(len(data['pmRA']))*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
#     st_coord = c.transform_to(mwsts[st].stream_frame)
    
    
#     phi1 = st_coord.phi1 #deg
#     phi2 = st_coord.phi2 #deg
#     pmphi1 = st_coord.pm_phi1_cosphi2 #mas/yr
#     pmphi2 = st_coord.pm_phi2 #mas/yr
#     d = data['Dist'] #kpc
#     e_d = d*0.03 #kpc
    
    
#     #Correcion por reflejo del sol (por ahora no lo uso)
#     c_reflex = gc.reflex_correct(c)
#     st_coord_reflex = c_reflex.transform_to(mwsts[st].stream_frame)
    
#     pmphi1_reflex = st_coord_reflex.pm_phi1_cosphi2 #mas/yr
#     pmphi2_reflex = st_coord_reflex.pm_phi2 #mas/yr
    
    
#     #Seleciono estrellas fuera del track
#     ra = data['RA_ICRS'] #deg
#     e_ra = data['e_RA_ICRS']/3600 #deg
#     dec = data['DE_ICRS'] #deg
#     e_dec = data['e_DE_ICRS']/3600 #deg
    
#     pmra = c.pm_ra_cosdec.value #data['pmRA'] #mas/yr
#     e_pmra = data['e_pmRA'] #mas/yr
#     pmdec = c.pm_dec.value #data['pmDE'] #mas/yr
#     e_pmdec = data['e_pmDE'] #mas/yr
    
    
#     #Create poly
#     field = ac.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    
#     skypath = np.loadtxt('pal5_extended_skypath.icrs.txt')
#     skypath_N = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,1]*u.deg, frame='icrs')
#     skypath_S = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,2]*u.deg, frame='icrs')

#     # Concatenate N track, S-flipped track and add first point at the end to close the polygon (needed for ADQL)
#     on_poly = ac.SkyCoord(ra = np.concatenate((skypath_N.ra,skypath_S.ra[::-1],skypath_N.ra[:1])),
#                             dec = np.concatenate((skypath_N.dec,skypath_S.dec[::-1],skypath_N.dec[:1])),
#                             unit=u.deg, frame='icrs')

#     #Select the field points inside the polygon footprint
#     footprint = galstreams.get_mask_in_poly_footprint(on_poly, field, stream_frame=mwsts[st].stream_frame)
#     off = ~footprint
    
#     # footprint = mwsts[st].get_mask_in_poly_footprint(field)
#     # off = ~mwsts[st].get_mask_in_poly_footprint(field)
    
#     ra_out = ra[off]
#     dec_out = dec[off]
#     e_ra_out = e_ra[off]
#     e_dec_out = e_dec[off]

#     pmra_out = pmra[off]
#     pmdec_out = pmdec[off]
#     e_pmra_out = e_pmra[off]
#     e_pmdec_out = e_pmdec[off]
    
#     d_out = d[off]
#     e_d_out = d_out*0.03
    
#     #Matriz de covarianza de las estrellas intrinsica + observacional
#     C_int = np.array([[C11, 0, 0], [0, C22, 0], [0, 0, C33]]) #Matriz de covarianza intrinseca (el stream tiene un ancho distinto a 0, los datos se alejan de la media no solo por el error observacional, sino tambien xq es intrinsecamente disperso): Fija para todas las estrellas
    
#     C_pm = [None for n in range(len(e_pmra))]  
#     for i in range(len(e_pmra)):
#         C_pm[i] = np.array([[e_pmra[i]**2,0],[0,e_pmdec[i]**2]])

#     C_pm = np.array(C_pm)
#     C_pm = gc.transform_pm_cov(c, C_pm, mwsts[st].stream_frame) #Transformo matriz cov de pm al frame del stream

#     C_obs = [None for n in range(len(e_pmra))] #Matriz de covarianza observacional en el frame del stream
#     for i in range(len(e_pmra)):
#         C_obs[i] = np.zeros((3,3))
#         C_obs[i][:2,:2] = C_pm[i]
#         C_obs[i][2,2] = e_d[i]**2

#     C_tot = C_int + C_obs
    
    
#     #Valore medio y errores del cluster en phi1 y phi2
#     c_mean = ac.ICRS(ra=ra_mean*u.degree, dec=dec_mean*u.degree, distance=d_mean*u.kpc, pm_ra_cosdec=mura_mean*u.mas/u.yr, pm_dec=mudec_mean*u.mas/u.yr, radial_velocity=0*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
#     st_coord = c_mean.transform_to(mwsts[st].stream_frame)

#     mu1_mean = st_coord.pm_phi1_cosphi2.value #mas/yr
#     mu2_mean = st_coord.pm_phi2.value #mas/yr

#     C_mean = np.array([[(e_mura*10)**2, cov_mu*100], [cov_mu*100, (e_mudec*10)**2]])
#     C_mean = gc.transform_pm_cov(c_mean, C_mean, mwsts[st].stream_frame)
#     e_mu1 = np.sqrt(C_mean[0,0])
#     e_mu2 = np.sqrt(C_mean[1,1])
#     cov_mu = C_mean[0,1]

    
    
#     #Estrellas del track y del fondo
#     d_in = (d>d_inf) & (d<d_sup)
#     inside = (data['Track']==1)
#     out = (data['Track']==0)
#     miembro = inside & (data['Memb']>0.5)
    
#     if printTrack == 'yes':
#         fig=plt.figure(1,figsize=(12,6))
#         fig.subplots_adjust(wspace=0.3,hspace=0.34,top=0.9,bottom=0.17,left=0.11,right=0.97)
#         ax=fig.add_subplot(121)
#         ax.plot(ra[inside],dec[inside],'.',ms=5)
#         ax.plot(ra[out],dec[out],'.',ms=5)
#         ax.plot(ra[miembro],dec[miembro],'*',c='red',ms=10)
#         ax.set_xlabel('$\\alpha$ (°)')
#         ax.set_ylabel('$\delta$ (°)')
#         ax.set_xlim([max(ra), min(ra)])
#         ax.set_ylim([min(dec), max(dec)])

#         ax=fig.add_subplot(122)
#         ax.scatter(pmra[inside & d_in],pmdec[inside & d_in],s=5, label='in')
#         ax.scatter(pmra[out & d_in],pmdec[out & d_in],s=5, label='out')
#         ax.scatter(pmra[miembro & d_in],pmdec[miembro & d_in],s=50, marker='*',color='red', label='memb')
#         ax.set_xlabel('$\mu_\\alpha*$ ("/año)')
#         ax.set_ylabel('$\mu_\delta$ ("/año)')
#         ax.set_title('${} < d < {}$ kpc'.format(d_inf, d_sup))
#         ax.legend(frameon=False, ncol=3, handlelength=0.1)
#         ax.set_xlim([-5,1])
#         ax.set_ylim([-5,1]);


#         fig2=plt.figure(2,figsize=(12,8))
#         fig2.subplots_adjust(wspace=0.3,hspace=0.34,top=0.95,bottom=0.13,left=0.1,right=0.97)
#         ax2=fig2.add_subplot(221)
#         ax2.plot(phi1[inside],phi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],phi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,phi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],phi2[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\phi_2$ (°)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-3,5])

#         ax2=fig2.add_subplot(222)
#         ax2.plot(phi1[inside],d[inside],'.',ms=5)
#         ax2.plot(phi1[out],d[out],'.',ms=2.5)
#         ax2.plot(phi1[miembro],d[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$d$ (kpc)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([13,25])

#         ax2=fig2.add_subplot(223)
#         ax2.plot(phi1[inside],pmphi1[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi1[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi1_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi1[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_1}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([1,6])

#         ax2=fig2.add_subplot(224)
#         ax2.plot(phi1[inside],pmphi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi2[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_2}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-2.5,2.5]);


#         fig3=plt.figure(3,figsize=(8,6))
#         fig3.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
#         ax3=fig3.add_subplot(111)
#         ax3.plot(ra[inside],dec[inside],'.',ms=5)
#         ax3.plot(ra[out],dec[out],'.',ms=2.5)
#         ax3.plot(track.ra, track.dec, 'k.', ms=1.)
#         ax3.plot(mwsts[st].poly_sc.icrs.ra, mwsts[st].poly_sc.icrs.dec, lw=1.,ls='--', color='black')
#         ax3.set_xlabel('$\\alpha$ (°)')
#         ax3.set_ylabel('$\delta$ (°)')
#         ax3.set_xlim([max(data['RA_ICRS']), min(data['RA_ICRS'])])
#         ax3.set_ylim([min(data['DE_ICRS']), max(data['DE_ICRS'])]);


#         fig.savefig('sample.png')
#         fig2.savefig('track_memb.png')
#         # fig3.savefig('track.png')

#     return data, phi1, phi2, pmphi1, pmphi2, pmphi1_reflex, pmphi2_reflex, pmra, pmdec, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, mu1_mean, mu2_mean, e_mu1, e_mu2, cov_mu, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out, C_tot, footprint






# def datos_consgr(tabla, st, printTrack, C11, C22, C33, d_mean, ra_mean, dec_mean, mura_mean, mudec_mean, e_mura, e_mudec, cov_mu, d_inf, d_sup):
#     """
#     Funcion que devuelve la posicion, movimientos propios y distancia de las estrellas y del track en el frame de la corriente, y los movimientos propios en ar y dec y distancia de las estrellas por fuera del track junto a sus errores
    
#     Inputs:
#     tabla: Nombre de la tabla donde se encuentran los datos crudos
#     st: Nombre de la corriente estelar
#     printTrack: Imprime (yes/no) el track en ar y dec, pm_ra y pm_dec, phi1 y phi2, phi1 y d, phi1 y pmphi1, phi1 y pmphi2
#     C11, C22, C33: Valores de la matriz de covarianza intriseca del stream en movimientos propios y distancia
#     ra_mean, dec_mean: Ascención recta y declinacion medios del stream
#     mura_mean, mura_mean: Movimientos propios medios en ascención recta y declinacion 
#     e_mura, e_mura, cov_mu: Errores de los movimientos propios medios en ascencion recta y declinacion, y el factor de correlacion entre ambos
#     d_inf: Distancia minima de la corriente
#     d_sup: Distancia maxima de la corriente
    
#     Outputs:
#     data: Tabla con los datos originales
#     phi1, phi2: Posicion de las estrellas en el frame de la corriente
#     pmphi1, pmphi2: Moviemientos propios de las estrellas en el frame de la corriente
#     pmphi1_reflex, pmphi2_reflex: Moviemientos propios de las estrellas en el frame de la corriente corregidas por el reflejo solar
#     pmra, pmdec, d: Movimientos propios en ascención recta y declinacion y distancias de las estrellas
#     phi1_t, phi2_t, pmphi1_t, pmphi2_t: Posicion y movimientos propios del track en el frame de la corriente
#     pmra_out, pmdec_out, d_out: movimientos propios y distancia de las estrellas fuera del track en ar y dec
#     e_pmra_out, e_pmdec_out, e_d_out: Errores en los movimientos propios y distancia de las estrellas fuera del track en ar y dec
#     C_tot: Matriz de covarianza intrinseca (fija) + observacional en el frame del stream
#     footprint: Mascara con las estrellas espacialmente dentro del track
#     """

#     print('\nCargando datos \n')

#     ##Cargo datos
#     f = fits.open(tabla)
#     data = f[1].data
#     # data.columns
#     # Table(data)

#     print('Cargando track y transformando coordenadas \n')
    
#     ##Cargo track y transformo coordenadas
#     mwsts = galstreams.MWStreams(verbose=False, implement_Off=False)
#     track = mwsts[st].track
#     #track = gc.reflex_correct(track) #Por ahora no uso la correcion por reflejo del sol
#     st_track = track.transform_to(mwsts[st].stream_frame)
    
#     phi1_t = st_track.phi1
#     phi2_t = st_track.phi2
#     pmphi1_t = st_track.pm_phi1_cosphi2
#     pmphi2_t = st_track.pm_phi2
    
        
#     _ = ac.galactocentric_frame_defaults.set('v4.0') #set the default Astropy Galactocentric frame parameters to the values adopted in Astropy v4.0
    
#     c = ac.ICRS(ra=data['RA_ICRS']*u.degree, dec=data['DE_ICRS']*u.degree, distance=data['Dist']*u.kpc, pm_ra_cosdec=data['pmRA']*u.mas/u.yr, pm_dec=data['pmDE']*u.mas/u.yr, radial_velocity=np.zeros(len(data['pmRA']))*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
#     st_coord = c.transform_to(mwsts[st].stream_frame)
    
    
#     phi1 = st_coord.phi1 #deg
#     phi2 = st_coord.phi2 #deg
#     pmphi1 = st_coord.pm_phi1_cosphi2 #mas/yr
#     pmphi2 = st_coord.pm_phi2 #mas/yr
#     d = data['Dist'] #kpc
#     e_d = d*0.03 #kpc
    
    
#     #Correcion por reflejo del sol (por ahora no lo uso)
#     c_reflex = gc.reflex_correct(c)
#     st_coord_reflex = c_reflex.transform_to(mwsts[st].stream_frame)
    
#     pmphi1_reflex = st_coord_reflex.pm_phi1_cosphi2 #mas/yr
#     pmphi2_reflex = st_coord_reflex.pm_phi2 #mas/yr
    
    
#     #Seleciono estrellas fuera del track
#     ra = data['RA_ICRS'] #deg
#     e_ra = data['e_RA_ICRS']/3600 #deg
#     dec = data['DE_ICRS'] #deg
#     e_dec = data['e_DE_ICRS']/3600 #deg
    
#     pmra = c.pm_ra_cosdec.value #data['pmRA'] #mas/yr
#     e_pmra = data['e_pmRA'] #mas/yr
#     pmdec = c.pm_dec.value #data['pmDE'] #mas/yr
#     e_pmdec = data['e_pmDE'] #mas/yr
    
    
#     #Create poly
#     field = ac.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    
#     skypath = np.loadtxt('pal5_extended_skypath.icrs.txt')
#     skypath_N = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,1]*u.deg, frame='icrs')
#     skypath_S = ac.SkyCoord(ra=skypath[:,0]*u.deg, dec=skypath[:,2]*u.deg, frame='icrs')

#     # Concatenate N track, S-flipped track and add first point at the end to close the polygon (needed for ADQL)
#     on_poly = ac.SkyCoord(ra = np.concatenate((skypath_N.ra,skypath_S.ra[::-1],skypath_N.ra[:1])),
#                             dec = np.concatenate((skypath_N.dec,skypath_S.dec[::-1],skypath_N.dec[:1])),
#                             unit=u.deg, frame='icrs')

#     #Select the field points inside the polygon footprint
#     footprint = galstreams.get_mask_in_poly_footprint(on_poly, field, stream_frame=mwsts[st].stream_frame)
#     off = ~footprint
    
#     # footprint = mwsts[st].get_mask_in_poly_footprint(field)
#     # off = ~mwsts[st].get_mask_in_poly_footprint(field)
    
#     ra_out = ra[off]
#     dec_out = dec[off]
#     e_ra_out = e_ra[off]
#     e_dec_out = e_dec[off]

#     pmra_out = pmra[off]
#     pmdec_out = pmdec[off]
#     e_pmra_out = e_pmra[off]
#     e_pmdec_out = e_pmdec[off]
    
#     d_out = d[off]
#     e_d_out = d_out*0.03
    
#     #Matriz de covarianza de las estrellas intrinsica + observacional
#     C_int = np.array([[C11, 0, 0], [0, C22, 0], [0, 0, C33]]) #Matriz de covarianza intrinseca (el stream tiene un ancho distinto a 0, los datos se alejan de la media no solo por el error observacional, sino tambien xq es intrinsecamente disperso): Fija para todas las estrellas
    
#     C_pm = [None for n in range(len(e_pmra))]  
#     for i in range(len(e_pmra)):
#         C_pm[i] = np.array([[e_pmra[i]**2,0],[0,e_pmdec[i]**2]])

#     C_pm = np.array(C_pm)
#     C_pm = gc.transform_pm_cov(c, C_pm, mwsts[st].stream_frame) #Transformo matriz cov de pm al frame del stream

#     C_obs = [None for n in range(len(e_pmra))] #Matriz de covarianza observacional en el frame del stream
#     for i in range(len(e_pmra)):
#         C_obs[i] = np.zeros((3,3))
#         C_obs[i][:2,:2] = C_pm[i]
#         C_obs[i][2,2] = e_d[i]**2

#     C_tot = C_int + C_obs
    
    
#     #Valore medio y errores del cluster en phi1 y phi2
#     c_mean = ac.ICRS(ra=ra_mean*u.degree, dec=dec_mean*u.degree, distance=d_mean*u.kpc, pm_ra_cosdec=mura_mean*u.mas/u.yr, pm_dec=mudec_mean*u.mas/u.yr, radial_velocity=0*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
#     st_coord = c_mean.transform_to(mwsts[st].stream_frame)

#     mu1_mean = st_coord.pm_phi1_cosphi2.value #mas/yr
#     mu2_mean = st_coord.pm_phi2.value #mas/yr

#     C_mean = np.array([[(e_mura*10)**2, cov_mu*100], [cov_mu*100, (e_mudec*10)**2]])
#     C_mean = gc.transform_pm_cov(c_mean, C_mean, mwsts[st].stream_frame)
#     e_mu1 = np.sqrt(C_mean[0,0])
#     e_mu2 = np.sqrt(C_mean[1,1])
#     cov_mu = C_mean[0,1]

    
    
#     #Estrellas del track y del fondo
#     d_in = (d>d_inf) & (d<d_sup)
#     inside = (data['Track']==1)
#     out = (data['Track']==0)
#     miembro = inside & (data['Memb']>0.5)
    
#     if printTrack == 'yes':
#         fig=plt.figure(1,figsize=(12,6))
#         fig.subplots_adjust(wspace=0.3,hspace=0.34,top=0.9,bottom=0.17,left=0.11,right=0.97)
#         ax=fig.add_subplot(121)
#         ax.plot(ra[inside],dec[inside],'.',ms=5)
#         ax.plot(ra[out],dec[out],'.',ms=5)
#         ax.plot(ra[miembro],dec[miembro],'*',c='red',ms=10)
#         ax.set_xlabel('$\\alpha$ (°)')
#         ax.set_ylabel('$\delta$ (°)')
#         ax.set_xlim([max(ra), min(ra)])
#         ax.set_ylim([min(dec), max(dec)])

#         ax=fig.add_subplot(122)
#         ax.scatter(pmra[inside & d_in],pmdec[inside & d_in],s=5, label='in')
#         ax.scatter(pmra[out & d_in],pmdec[out & d_in],s=5, label='out')
#         ax.scatter(pmra[miembro & d_in],pmdec[miembro & d_in],s=50, marker='*',color='red', label='memb')
#         ax.set_xlabel('$\mu_\\alpha*$ ("/año)')
#         ax.set_ylabel('$\mu_\delta$ ("/año)')
#         ax.set_title('${} < d < {}$ kpc'.format(d_inf, d_sup))
#         ax.legend(frameon=False, ncol=3, handlelength=0.1)
#         ax.set_xlim([-5,1])
#         ax.set_ylim([-5,1]);


#         fig2=plt.figure(2,figsize=(12,8))
#         fig2.subplots_adjust(wspace=0.3,hspace=0.34,top=0.95,bottom=0.13,left=0.1,right=0.97)
#         ax2=fig2.add_subplot(221)
#         ax2.plot(phi1[inside],phi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],phi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,phi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],phi2[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\phi_2$ (°)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-3,5])

#         ax2=fig2.add_subplot(222)
#         ax2.plot(phi1[inside],d[inside],'.',ms=5)
#         ax2.plot(phi1[out],d[out],'.',ms=2.5)
#         ax2.plot(phi1[miembro],d[miembro],'*',c='red',ms=10.)
#         # ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$d$ (kpc)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([13,25])

#         ax2=fig2.add_subplot(223)
#         ax2.plot(phi1[inside],pmphi1[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi1[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi1_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi1[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_1}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([1,6])

#         ax2=fig2.add_subplot(224)
#         ax2.plot(phi1[inside],pmphi2[inside],'.',ms=5)
#         ax2.plot(phi1[out],pmphi2[out],'.',ms=2.5)
#         ax2.plot(phi1_t,pmphi2_t,'k.',ms=1.)
#         ax2.plot(phi1[miembro],pmphi2[miembro],'*',c='red',ms=10.)
#         ax2.set_xlabel('$\phi_1$ (°)')
#         ax2.set_ylabel('$\mu_{\phi_2}$ ("/año)')
#         ax2.set_xlim([-20,15])
#         ax2.set_ylim([-2.5,2.5]);


#         fig3=plt.figure(3,figsize=(8,6))
#         fig3.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
#         ax3=fig3.add_subplot(111)
#         ax3.plot(ra[inside],dec[inside],'.',ms=5)
#         ax3.plot(ra[out],dec[out],'.',ms=2.5)
#         ax3.plot(track.ra, track.dec, 'k.', ms=1.)
#         ax3.plot(mwsts[st].poly_sc.icrs.ra, mwsts[st].poly_sc.icrs.dec, lw=1.,ls='--', color='black')
#         ax3.set_xlabel('$\\alpha$ (°)')
#         ax3.set_ylabel('$\delta$ (°)')
#         ax3.set_xlim([max(data['RA_ICRS']), min(data['RA_ICRS'])])
#         ax3.set_ylim([min(data['DE_ICRS']), max(data['DE_ICRS'])]);


#         fig.savefig('sample.png')
#         fig2.savefig('track_memb.png')
#         # fig3.savefig('track.png')

#     return data, phi1, phi2, pmphi1, pmphi2, pmphi1_reflex, pmphi2_reflex, pmra, pmdec, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, mu1_mean, mu2_mean, e_mu1, e_mu2, cov_mu, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out, C_tot, footprint
