##############################################
# SYNCMD Creation
##############################################
#
# Processing Steps:
# make_specgrid
# make_sedgrid
#
# Notes
# 1)Import assumptions (filters, avg DM) are set in DATAMODEL ITEMS block
# 2) Code defaults to overwriting output files
#
# Example
# import run_syncmd
# run_syncmd.make_specgrid(specfile='syncmd_spec_hi.grid.hd5')
# run_syncmd.make_sedgrid(sedfile='syncmd_sedsobs.fits',
#                         specfile='syncmd_spec_hi.grid.hd5')
#
##############################################

import numpy as np
import scipy
import scipy.stats
import os
import tables
from astropy.table import Table as apyTable

from beast.external.eztables import Table
from astropy import units
from beast.physicsmodel import creategrid
from beast.physicsmodel.grid import SpectralGrid
from beast.physicsmodel.stars import stellib
from beast.physicsmodel.dust import extinction
from beast.observationmodel import phot
from beast.observationmodel.vega import Vega
import beast.observationmodel.noisemodel.generic_noisemodel as noisemodel

# DATAMODEL ITEMS
filters = ['HST_WFC3_F225W','HST_WFC3_F275W','HST_WFC3_F336W',
           'HST_ACS_WFC_F475W','HST_ACS_WFC_F550M','HST_ACS_WFC_F658N',
           'HST_ACS_WFC_F814W','HST_WFC3_F110W','HST_WFC3_F160W']
additional_filters = ['GALEX_FUV', 'GALEX_NUV']
add_spectral_properties_kwargs = dict(filternames=filters + additional_filters)


def make_specgrid(specfile='syncmd_spec.grid.hd5',
                  fakein='syncmd_final-loz_parsec.fits',
                  distanceModulus=18.96, zsol=0.0142,
                  trimspec=False, grngspec=[1.15e3,3.0e4],
                  use_btsettl=False, btsettl_medres=False):
    """
    Create spectral grid from FAKE output

    Parameters
    ----------

    specfile: str
        file into which save the spectral grid; format = .grid.hd5

    fakein: str
        output file from FAKE used as input

    """

    idistanceModulus = distanceModulus * units.mag
    dmod = idistanceModulus.to(units.mag).value
    distance = 10 ** ( (dmod / 5.) + 1 ) * units.pc

    if use_btsettl:
        osl = stellib.BTSettl(medres=btsettl_medres)
    else:
        osl = stellib.Tlusty() + stellib.Kurucz()

    synraw = apyTable.read(fakein)

    synin = Table()
    synin.addCol('logg', synraw['MLOGG']*1.0)
    synin.addCol('logT', synraw['MLOGT']*1.0)
    synin.addCol('logL', (-0.4)*(synraw['MMBOL']-distanceModulus-4.77))
    synin.addCol('Z', 10.**(synraw['MHZ'])*zsol)
    synin.addCol('logA', np.log10(synraw['AGE'])+9.0)
    synin.addCol('M_ini', synraw['MMASS']*1.0)

    spgrid = osl.gen_spectral_grid_from_given_points(synin)

    _distance = distance.to(units.pc).value
    spgrid.seds = spgrid.seds / (0.1 * _distance) ** 2   # Convert from 10 pc
    #nameformat = add_spectral_properties_kwargs.pop('nameformat', '{0:s}') + '_nd'
    #spgrid = creategrid.add_spectral_properties(spgrid, nameformat=nameformat,
    #                                            **add_spectral_properties_kwargs)

    # Trim spec for good extLaw range
    if trimspec:
        sel = ((spgrid.lamb > grngspec[0]) & (spgrid.lamb < grngspec[1]))
        spgrid.lamb=spgrid.lamb[sel]
        spgrid.seds=spgrid.seds[:,sel]

    # Write out file, remove if it exists
    try:
        os.remove(specfile)
    except OSError:
        pass
    spgrid.writeHDF(specfile)


def make_sedgrid_av0(sedav0file=None, sedav0filegrid='syncmd_sedsav0.grid.hd5',
                     specfile='syncmd_spec.grid.hd5', distanceModulus=18.96):
    """
    Create raw Av=0 SED grid from spectral grid and write output to FITS file.

    Parameters
    ----------

    sedav0file: str
        output file for observed SEDs; format = .fits
        default=None; no FITS file written unless param is passed

    sedav0filegrid: str
        output file for observed SEDs; format = .grid.hd5;

    specfile: str
        input file from make_specgrid; format = .grid.hd5

    """

    # Load spec grid
    spgrid = SpectralGrid(specfile, backend='memory')
    N = len(spgrid.grid)

    # Compute Vega Fluxes
    _, vega_flux, _ = Vega().getFlux(filters)

    # Compute SEDs
    cols = {}
    keys = spgrid.keys()
    for key in keys:
        cols[key] = np.empty(N, dtype=float)

    #nameformat = add_spectral_properties_kwargs.pop('nameformat','{0:s}') + '_wd'
    #spgrid = creategrid.add_spectral_properties(spgrid, nameformat=nameformat,
    #                                            **add_spectral_properties_kwargs)

    sed_results = spgrid.getSEDs(filters)
    _lamb = sed_results.lamb[:]
    #_seds = ((-2.5)*np.log10(sed_results.seds[:]/vega_flux))
    _seds = sed_results.seds[:]

    for key in sed_results.grid.keys():
        if key not in keys:
            cols[key] = np.empty(N, dtype=float)
            cols[key] = sed_results.grid[key]

    # copy the rest of the parameters
    for key in keys:
        cols[key] = spgrid.grid[key]

    g = SpectralGrid(_lamb, seds=_seds, grid=Table(cols), backend='memory')
    g.grid.header['filters'] = ' '.join(filters)
    g.grid.header['dmod'] = distanceModulus
    g.grid.header['specfile'] = specfile

    # Write out HD5 SED file if param given, remove if it exists
    if sedav0filegrid is None:
        pass
    else:
        try:
            os.remove(sedav0filegrid)
        except OSError:
            pass
        g.writeHDF(sedav0filegrid)

    if sedav0file is None:
        pass
    else:
        mag_av0 = ((-2.5)*np.log10(sed_results.seds[:]/vega_flux))

        # Prep FITS Table
        filters_av0 = []
        for f in filters:
            filters_av0.append(f.split('_')[-1].upper() + '_ORIG')

        data = apyTable()
        for i, f in enumerate(filters_av0):
            data[f] = mag_av0[:,i]

        data['logA'] = g['logA']
        data['M_ini'] = g['M_ini']
        data['Z'] = g['Z']

        # Header Info
        data.meta['dmod'] = distanceModulus
        data.meta['specfile'] = specfile

        # Write FITS file, remove if it exists
        data.write(sedav0file, overwrite=True)

def make_sedgrid_red(sedfile='syncmd_sedsobs.fits', sedfilegrid=None,
                     sedav0filegrid='syncmd_sedsav0.grid.hd5',
                     astfile='ast_half1+3_wbg.fits',
                     att_coeff=[2.49,2.37,1.97,1.27,1.07,0.881,0.652,0.370,0.222],
                     att_coeff_mw=[2.33,1.94,1.66,1.19,0.98,0.81,0.606,0.337,0.204],
                     av_fg=0.18, av_red_median=0.4, av_red_loc=0.0,
                     av_red_sig=0.55, av_unred_max=0.0, dmod_sig_old=0.15,
                     dust_dmod_relative=-0.1, sclh_ratio_max=10.,
                     sclh_ratio_min=1.,sclh_loga_transition=8.5,
                     output_raw_cols=False, output_allraw_cols=False,
                     distanceModulus=18.96):
    """
    Create SED grid from SED_av0 grid, applying dust attenuation and
    distance shifts.  Write output SEDs into a FITS file.

    Model includes age-dependent extinction, implemented as a simple two
    component model (young stars, old stars; divided at age defined by
    sclh_loga_transition) where variables are linked:
    1) dmod_sig_old sets maximum DM, 2) dmod_sig_dust set by dmod_sig_old &
    sclh_ratio_max, 3) dmod_sig_yng set by dmod_sig_dust & sclh_ratio_min

    Parameters
    ----------

    sedfile: str
        output file for observed SEDs; format = .fits
    sedfilegrid: str
        output file for observed SEDs; format = .grid.hd5;
        default=None; no grid file written unless param is passed
    sedav0filegrid: str
        input file for Av=0 SEDs; format = .grid.hd5;
    astfile: str
        input file for ASTs; format = .fits
    att_coeff: list
        list of dust attenuation coefficients; one per filter
    av_fg: float
        foreground (MW) Av in magnitudes; default = 0.1 mag
    av_red_median: float
        median of lognormal dist. for Av in magnitudes; where
        av_red_mean = av_red_median * exp(av_red_sig**2./2.0); default = 0.5 mag
    av_red_loc: floag
        zeropoint of lognormal dist.; default = 0.0 mag
    av_red_sig: float
        sigma of lognormal dist. for Av in magnitudes; default = 0.5 mag
    av_unred_max: float
        maximum Av for uniform unreddened dist. magnitudes; default = 0.1 mag
    distanceModulus: float
        distance modulus used for grid, passed into header
    dmod_sig_old: float
        sigma of normal dist. (centered at 0.) of distance modulus offsets,
        where offsets are relative to mean set in preamble; default=0.15 mag
    dust_dmod_relative: float
        offset of dust from average distance, given in mag w.r.t. average
        distance modulus; default=-0.05 mag

    sclh_ratio_max: float
        for step-function scale height model, this is large value adopted at
        old ages when dust is in thin plane with respect to dust; default = 10.
    sclh_ratio_min: float
        for step function scale height model, this is small value adopted at
        young ages when stars and dust are well-mixed; default = 1.
    sclh_loga_transition: float
        log(age/yr) of step-function transition point for scale height
        difference; default = 8.5

    output_raw_cols: boolean
        flag to add RAW and ORIG columns to output file
    output_allraw_cols: boolean
        flag to add RAW_AV and RAW_DM columns to output file

    """

    # Load spec grid
    g = SpectralGrid(sedav0filegrid, backend='memory')
    N = len(g.grid)

    # Compute Vega Fluxes
    _, vega_flux, _ = Vega().getFlux(filters)

    # Compute Orig Fluxes + Mags (w/o Av + Dmod Shifts)
    mag_av0 = ((-2.5)*np.log10(g.seds[:]/vega_flux))

    ### Set Distance Modulus Distribution

    # Calc Constants
    dmod_sig_dust = dmod_sig_old / sclh_ratio_max
    dmod_sig_yng = dmod_sig_dust * sclh_ratio_min

    # Current: Normal w/ sigma=dmod_sig
    dmod_offset_raw = scipy.random.normal(0.,1.0,N)
    # Add logic for assigning scalings -- current: step function
    idmod_sig = np.zeros(N)
    idmod_sig[g['logA'] < sclh_loga_transition] = dmod_sig_yng
    idmod_sig[g['logA'] >= sclh_loga_transition] = dmod_sig_old
    idmod_off = np.zeros(N)
    idmod_off[g['logA'] < sclh_loga_transition] = dust_dmod_relative
    #idmod_off[g['logA'] < sclh_loga_transition] = 0.0
    dmod_offset = (dmod_offset_raw * idmod_sig) + idmod_off

    # Set Av Distribution
    # Current: Lognormal w/ median=av_red_median, sigma=av_red_sig
    #   -Dust Pos = dust_dmod_relative, sets f_red
    #   -Foreground Pop = Uniform from Av=0-av_unred_max
    #   -MW Foreground = av_fg added to all sources
    av_draw = scipy.stats.lognorm.rvs(av_red_sig,loc=av_red_loc,
                                      scale=av_red_median,size=N)
    #av[np.where(av < 0.0)] = 0.0 #Clip negative Av tail

    # Assign Av via Z distribution
    z_erf = (dmod_offset-dust_dmod_relative)/dmod_sig_dust
    av = av_draw * 0.5*(1.+scipy.special.erf(z_erf/np.sqrt(2.)))
    f_red = -99.99

    # Foreground Pop
    #fgpop, = np.where(dmod_offset < dust_dmod_relative)
    #n_fgpop = len(fgpop)
    #av[fgpop] = scipy.random.uniform(0.0,av_unred_max,n_fgpop)
    #f_red = 1.-(n_fgpop/float(N))
    #print('f_red = {:5.3f}'.format(f_red))

    # Add Foreground Reddening
    av_tot = av + av_fg

    ###########################################

    # Redden SEDs
    g.seds *= 10.**(-0.4*(av[:,np.newaxis] * np.array(att_coeff)))
    g.seds *= 10.**(-0.4*(av_fg * np.array(att_coeff_mw)))

    flux_avonly = g.seds[:].copy()
    mag_raw_av = ((-2.5)*np.log10(flux_avonly/vega_flux))

    # Add Distance Offset
    g.seds *= 10.**(-0.4*dmod_offset[:,np.newaxis])

    mag_raw_dm = mag_av0.copy() + dmod_offset[:,np.newaxis]

    # Compute SEDs
    cols = {'Av': np.empty(N, dtype=float), 'Dmod_offset': np.empty(N, dtype=float)}
    keys = g.keys()
    for key in keys:
        cols[key] = np.empty(N, dtype=float)
    cols['Av'] = av_tot
    cols['Dmod_offset'] = dmod_offset

    #nameformat = add_spectral_properties_kwargs.pop('nameformat','{0:s}') + '_wd'
    #gout = creategrid.add_spectral_properties(gout, nameformat=nameformat,
    #                                            **add_spectral_properties_kwargs)

    _lamb = g.lamb[:]
    _seds = ((-2.5)*np.log10(g.seds[:]/vega_flux))

    for key in g.grid.keys():
        if key not in keys:
            cols[key] = np.empty(N, dtype=float)
            cols[key] = g.grid[key]

    # copy the rest of the parameters
    for key in keys:
        cols[key] = g.grid[key]

    gout = SpectralGrid(_lamb, seds=_seds, grid=Table(cols), backend='memory')
    gout.grid.header['filters'] = ' '.join(filters)
    gout.grid.header['av_fg'] = av_fg
    gout.grid.header['av_red_median'] = av_red_median
    gout.grid.header['av_red_loc'] = av_red_loc
    gout.grid.header['av_red_sig'] = av_red_sig
    gout.grid.header['av_unred_max'] = av_unred_max
    gout.grid.header['dmod'] = distanceModulus
    gout.grid.header['dmod_sig_old'] = dmod_sig_old
    gout.grid.header['dmod_sig_yng'] = dmod_sig_yng
    gout.grid.header['sclh_loga_transition'] = sclh_loga_transition
    gout.grid.header['dust_dmod_relative'] = dust_dmod_relative
    gout.grid.header['f_red'] = f_red
    gout.grid.header['extlaw'] = 'Att Coeff'
    gout.grid.header['specfile'] = g.grid.header['specfile']
    gout.grid.header['astfile'] = astfile

    ###########################################

    # Add Observational Noise + Completeness
    mag_raw = gout.seds[:].copy()

    flux = g.seds[:]
    N, M = flux.shape

    model = noisemodel.Generic_ToothPick_Noisemodel(astfile, filters)
    model.fit_bins(nbins=30, completeness_mag_cut=80)

    bias = np.empty((N, M), dtype=float)
    sigma = np.empty((N, M), dtype=float)
    compl = np.empty((N, M), dtype=float)
    flux_out = np.empty((N, M), dtype=float)
    mag_out = np.empty((N, M), dtype=float)
    mag_out_obs = np.empty((N, M), dtype=float)

    for i in range(M):
        ncurasts = model._nasts[i]
        _fluxes = model._fluxes[0:ncurasts, i]
        _biases = model._biases[0:ncurasts, i]
        _sigmas = model._sigmas[0:ncurasts, i]
        _compls = model._compls[0:ncurasts, i]

        arg_sort = np.argsort(_fluxes)
        _fluxes = _fluxes[arg_sort]

        bias[:, i] = np.interp(flux[:, i], _fluxes, _biases[arg_sort] )
        sigma[:, i] = np.interp(flux[:, i], _fluxes, _sigmas[arg_sort])
        compl[:, i] = np.interp(flux[:, i], _fluxes, _compls[arg_sort])

        dlt_flux = scipy.random.normal(size=N)
        flux_out[:, i] = flux[:,i]+bias[:,i]+(dlt_flux*sigma[:,i])
        #flux_out[(flux_out[:,i] < 0.), i] = 0. # TODO: set floor at 0
        mag_out[:, i] = (-2.5)*np.log10(flux_out[:,i]/vega_flux[i])
        mag_out_obs[:, i] = (-2.5)*np.log10(flux_out[:,i]/vega_flux[i])

        draw_comp = scipy.random.uniform(size=N)
        # DETECTION CHOICE: based on draw_comp, option of 0.5 hard cut
        #nondetect, = np.where((compl[:,i] < draw_comp) | (compl[:,i] < 0.5))
        nondetect, = np.where((compl[:,i] < draw_comp))
        mag_out_obs[nondetect, i] = np.nan

    gout.seds[:] = mag_out_obs

    # Write out HD5 SED file if param given, remove if it exists
    if sedfilegrid is None:
        pass
    else:
        try:
            os.remove(sedfilegrid)
        except OSError:
            pass
        gout.writeHDF(sedfilegrid)

    # Prep FITS Table
    filters_syn = []
    filters_raw = []
    filters_av0 = []
    for f in filters:
        filters_syn.append(f.split('_')[-1].upper() + '_SYN')
        filters_raw.append(f.split('_')[-1].upper() + '_RAW')
        filters_av0.append(f.split('_')[-1].upper() + '_ORIG')

    data = apyTable()
    for i, f in enumerate(filters_syn):
        data[f] = mag_out_obs[:,i]

    if output_raw_cols:
        for i, f in enumerate(filters_raw):
            data[f] = mag_raw[:,i]
        for i, f in enumerate(filters_av0):
            data[f] = mag_av0[:,i]

    if output_allraw_cols:
        for i, f in enumerate(filters_raw):
            data[f+'_AV'] = mag_raw_av[:,i]
            data[f+'_DM'] = mag_raw_dm[:,i]

    data['Av'] = gout['Av']
    data['Dmod_offset'] = gout['Dmod_offset']
    data['logA'] = gout['logA']
    data['M_ini'] = gout['M_ini']
    data['Z'] = gout['Z']

    # Header Info
    data.meta['av_fg'] = av_fg
    data.meta['av1_med'] = av_red_median
    data.meta['av1_sig'] = av_red_sig
    data.meta['av0_max'] = av_unred_max
    data.meta['dmod'] = distanceModulus
    data.meta['dsig_old'] = dmod_sig_old
    data.meta['dsig_yng'] = dmod_sig_old
    data.meta['sclhloga'] = sclh_loga_transition
    data.meta['dmod_rel'] = dust_dmod_relative
    data.meta['f_red'] = f_red
    data.meta['extlaw'] = 'Att Coeff'
    data.meta['specfile'] = g.grid.header['specfile']
    data.meta['astfile'] = astfile

    # Write FITS file, remove if it exists
    data.write(sedfile, overwrite=True)
