# from parametros import *
import numpy as np
import scipy
import pylab as plt
import seaborn as sns
sns.set(style="ticks", context="poster")

import astropy
from astropy.table import Table
import astropy.coordinates as ac
import astropy.units as u
from astropy.io import fits
import gala.coordinates as gc
import galstreams


def datos(tabla, st, printTrack, C11, C22, C33, d_inf, d_sup):
    """
    Funcion que devuelve la posicion, movimientos propios y distancia de las estrellas y del track en el frame de la corriente, y los movimientos propios en ar y dec y distancia de las estrellas por fuera del track junto a sus errores
    
    Inputs:
    tabla: Nombre de la tabla donde se encuentran los datos crudos
    st: Nombre de la corriente estelar
    printTrack: Imprime (yes/no) el track en ar y dec, pm_ra y pm_dec, phi1 y phi2, phi1 y d, phi1 y pmphi1, phi1 y pmphi2
    d_inf: Distancia minima de la corriente
    d_sup: Distancia maxima de la corriente
    
    Outputs:
    data: Tabla con los datos originales
    phi1, phi2: Posicion de las estrellas en el frame de la corriente
    pmphi1, pmphi2: Moviemientos propios de las estrellas en el frame de la corriente
    pmra, pmdec, d: Movimientos propios en ascención recta y declinacion y distancias de las estrellas
    phi1_t, phi2_t, pmphi1_t, pmphi2_t: Posicion y movimientos propios del track en el frame de la corriente
    pmra_out, pmdec_out, d_out: movimientos propios y distancia de las estrellas fuera del track en ar y dec
    e_pmra_out, e_pmdec_out, e_d_out: Errores en los movimientos propios y distancia de las estrellas fuera del track en ar y dec
    footprint: Mascara con las estrellas espacialmente dentro del track
    """

    print('\nCargando datos \n')

    ##Cargo datos
    f = fits.open(tabla)
    data = f[1].data
    # data.columns


    print('Cargando track y transformando coordenadas \n')
    
    ##Cargo track y transformo coordenadas
    mwsts = galstreams.MWStreams(verbose=False, implement_Off=False)
    track = mwsts[st].track
    track = gc.reflex_correct(track)
    st_track = track.transform_to(mwsts[st].stream_frame)
    
    phi1_t = st_track.phi1
    phi2_t = st_track.phi2
    pmphi1_t = st_track.pm_phi1_cosphi2
    pmphi2_t = st_track.pm_phi2
    
    _ = ac.galactocentric_frame_defaults.set('v4.0') #set the default Astropy Galactocentric frame parameters to the values adopted in Astropy v4.0
    
    c = ac.ICRS(ra=data['RA_ICRS']*u.degree, dec=data['DE_ICRS']*u.degree, distance=data['Dist']*u.kpc, pm_ra_cosdec=data['pmRA']*u.mas/u.yr, pm_dec=data['pmDE']*u.mas/u.yr, radial_velocity=np.zeros(len(data['pmRA']))*u.km/u.s) #The input coordinate instance must have distance and radial velocity information. So, if the radial velocity is not known, fill the radial velocity values with zeros to reflex-correct the proper motions.
    st_coord = c.transform_to(mwsts[st].stream_frame)
    
    c_reflex = gc.reflex_correct(c)
    st_coord_reflex = c_reflex.transform_to(mwsts[st].stream_frame)
    
    phi1 = st_coord.phi1 #deg
    phi2 = st_coord.phi2 #deg
    pmphi1 = st_coord.pm_phi1_cosphi2 #mas/yr
    pmphi2 = st_coord.pm_phi2 #mas/yr
    d = data['Dist'] #kpc
    e_d = d*0.03 #kpc
    
    #Seleciono estrellas fuera del track
    ra = data['RA_ICRS'] #deg
    e_ra = data['e_RA_ICRS']/3600 #deg
    dec = data['DE_ICRS'] #deg
    e_dec = data['e_DE_ICRS']/3600 #deg
    
    pmra = c.pm_ra_cosdec.value #data['pmRA'] #mas/yr
    e_pmra = data['e_pmRA'] #mas/yr
    pmdec = c.pm_dec.value #data['pmDE'] #mas/yr
    e_pmdec = data['e_pmDE'] #mas/yr
    
    pmphi1_reflex = st_coord_reflex.pm_phi1_cosphi2 #mas/yr
    pmphi2_reflex = st_coord_reflex.pm_phi2 #mas/yr
    
    field = ac.SkyCoord(ra=ra*u.deg, dec=dec*u.deg, frame='icrs')
    #Select the field points inside the polygon footprint
    footprint = mwsts[st].get_mask_in_poly_footprint(field)
    off = ~mwsts[st].get_mask_in_poly_footprint(field)
    
    ra_out = ra[off]
    dec_out = dec[off]
    e_ra_out = e_ra[off]
    e_dec_out = e_dec[off]

    pmra_out = pmra[off]
    pmdec_out = pmdec[off]
    e_pmra_out = e_pmra[off]
    e_pmdec_out = e_pmdec[off]
    
    d_out = d[off]
    e_d_out = d_out*0.03
    
    #Matriz de covarianza de las estrellas intrinsica+observacional
    C_int = np.array([[C11, 0, 0], [0, C22, 0], [0, 0, C33]]) #Matriz de covarianza intrinseca: Fija para todas las estrellas
    
    C_pm = [None for n in range(len(e_pmra))]  
    for i in range(len(e_pmra)):
        C_pm[i] = np.array([[e_pmra[i]**2,0],[0,e_pmdec[i]**2]])

    C_pm = np.array(C_pm)
    C_pm = gc.transform_pm_cov(c, C_pm, mwsts[st].stream_frame) #Transformo matriz cov de pm al frame del stream

    C_obs = [None for n in range(len(e_pmra))] #Matriz de covarianza observacional en el frame del stream
    for i in range(len(e_pmra)):
        C_obs[i] = np.zeros((3,3))
        C_obs[i][:2,:2] = C_pm[i]
        C_obs[i][2,2] = e_d[i]**2

    C_tot = C_int + C_obs
    
    #Estrellas del track y del fondo
    d_in = (data['Dist']>d_inf) & (data['Dist']<d_sup)
    inside = (data['Track']==1)
    out = (data['Track']==0)
    miembro = inside & (data['Memb']>0.5)
    
    if printTrack == 'yes':
        fig=plt.figure(1,figsize=(12,6))
        fig.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
        ax=fig.add_subplot(121)
        ax.plot(data[inside]['RA_ICRS'],data[inside]['DE_ICRS'],'.',ms=5)
        ax.plot(data[out]['RA_ICRS'],data[out]['DE_ICRS'],'.',ms=5)
        ax.plot(data[miembro]['RA_ICRS'],data[miembro]['DE_ICRS'],'*',c='red',ms=10)
        ax.set_xlabel('$\\alpha$ (°)')
        ax.set_ylabel('$\delta$ (°)')
        ax.set_xlim([max(data['RA_ICRS']), min(data['RA_ICRS'])])
        ax.set_ylim([min(data['DE_ICRS']), max(data['DE_ICRS'])])

        ax=fig.add_subplot(122)
        ax.scatter(data[inside & d_in]['pmRA'],data[inside & d_in]['pmDE'],s=5, label='in')
        ax.scatter(data[out & d_in]['pmRA'],data[out & d_in]['pmDE'],s=5, label='out')
        ax.scatter(data[miembro & d_in]['pmRA'],data[miembro & d_in]['pmDE'],s=50, marker='*',color='red', label='memb')
        ax.set_xlabel('$\mu_\\alpha*$ ("/año)')
        ax.set_ylabel('$\mu_\delta$ ("/año)')
        ax.set_title('${} < d < {}$ kpc'.format(d_inf, d_sup))
        ax.legend(frameon=False, ncol=3, handlelength=0.1)
        ax.set_xlim([-5,1])
        ax.set_ylim([-5,1]);


        fig2=plt.figure(2,figsize=(12,8))
        fig2.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
        ax2=fig2.add_subplot(221)
        ax2.plot(phi1[inside],phi2[inside],'.',ms=5)
        ax2.plot(phi1[out],phi2[out],'.',ms=2.5)
        ax2.plot(phi1_t,phi2_t,'k.',ms=1.)
        ax2.plot(phi1[miembro],phi2[miembro],'*',c='red',ms=10.)
        # ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$\phi_2$ (°)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([-3,5])

        ax2=fig2.add_subplot(222)
        ax2.plot(phi1[inside],d[inside],'.',ms=5)
        ax2.plot(phi1[out],d[out],'.',ms=2.5)
        ax2.plot(phi1[miembro],d[miembro],'*',c='red',ms=10.)
        # ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$d$ (kpc)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([13,25])

        ax2=fig2.add_subplot(223)
        ax2.plot(phi1[inside],pmphi1[inside],'.',ms=5)
        ax2.plot(phi1[out],pmphi1[out],'.',ms=2.5)
        ax2.plot(phi1_t,pmphi1_t,'k.',ms=1.)
        ax2.plot(phi1[miembro],pmphi1[miembro],'*',c='red',ms=10.)
        ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$\mu_{\phi_1}$ ("/año)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([1,6])

        ax2=fig2.add_subplot(224)
        ax2.plot(phi1[inside],pmphi2[inside],'.',ms=5)
        ax2.plot(phi1[out],pmphi2[out],'.',ms=2.5)
        ax2.plot(phi1_t,pmphi2_t,'k.',ms=1.)
        ax2.plot(phi1[miembro],pmphi2[miembro],'*',c='red',ms=10.)
        ax2.set_xlabel('$\phi_1$ (°)')
        ax2.set_ylabel('$\mu_{\phi_2}$ ("/año)')
        ax2.set_xlim([-20,15])
        ax2.set_ylim([-2.5,2.5]);


        fig3=plt.figure(3,figsize=(8,6))
        fig3.subplots_adjust(wspace=0.25,hspace=0.34,top=0.95,bottom=0.07,left=0.07,right=0.95)
        ax3=fig3.add_subplot(111)
        ax3.plot(data[inside]['RA_ICRS'],data[inside]['DE_ICRS'],'.',ms=5)
        ax3.plot(data[out]['RA_ICRS'],data[out]['DE_ICRS'],'.',ms=2.5)
        ax3.plot(track.ra, track.dec, 'k.', ms=1.)
        ax3.plot(mwsts[st].poly_sc.icrs.ra, mwsts[st].poly_sc.icrs.dec, lw=1.,ls='--', color='black')
        ax3.set_xlabel('$\\alpha$ (°)')
        ax3.set_ylabel('$\delta$ (°)')
        ax3.set_xlim([max(data['RA_ICRS']), min(data['RA_ICRS'])])
        ax3.set_ylim([min(data['DE_ICRS']), max(data['DE_ICRS'])]);


        fig.savefig('sample.png')
        fig2.savefig('track_memb.png')
        # fig3.savefig('track.png')

    return data, phi1, phi2, pmphi1, pmphi2, pmphi1_reflex, pmphi2_reflex, pmra, pmdec, d, phi1_t, phi2_t, pmphi1_t, pmphi2_t, pmra_out, pmdec_out, d_out, e_pmra_out, e_pmdec_out, e_d_out, C_tot, footprint

