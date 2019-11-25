from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, BrownianMotion2D
from datetime import timedelta as delta
from datetime import datetime
import numpy as np
import math
from glob import glob

withstokes = False
withwind = 0.01  # scaling_factor

ddir='/data/'

for startyear in np.arange(2000, 2014):
    print(startyear, withstokes, withwind)

    cmemsfiles = sorted(glob(ddir+'oceanparcels/input_data/CMEMS/GLOBAL_REANALYSIS_PHY_001_030/mercatorglorys12v1_gl12_mean_*.nc'))
    my = np.array([int(f[-21:-15]) for f in cmemsfiles])
    files = [cmemsfiles[i] for i in np.where(np.logical_and(startyear*100+8 <= my, my <= (startyear+2)*100+9))[0]]

    variables = {'U': 'uo', 'V': 'vo'}
    dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
    fset_currents = FieldSet.from_netcdf(files, variables, dimensions)
    fset_currents.add_periodic_halo(zonal=True)
    size2D = (fset_currents.U.grid.ydim, fset_currents.U.grid.xdim)

    fname = 'pumiceevent_aug_%d' %startyear

    if withstokes:
        stokesfiles = sorted(glob(ddir+'oceanparcels/input_data/WaveWatch3data/CFSR/WW3-GLOB-30M_*_uss.nc'))
        my = np.array([int(f[-13:-7]) for f in stokesfiles])
        stokesfiles = [stokesfiles[i] for i in np.where(np.logical_and(startyear*100+8 <= my, my <= (startyear+2)*100+9))[0]]
        stokesdimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
        stokesvariables = {'U': 'uuss', 'V': 'vuss'}
        fset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
        fset_stokes.add_periodic_halo(zonal=True, meridional=False, halosize=5)
        fname+='_wstokes'

    if withwind:
        windfiles = sorted(glob(ddir+'oceanparcels/input_data/CMEMS/CERSAT-GLO-BLENDED_WIND_L4_REP-V6-OBS_FULL_TIME_SERIE/*-IFR-L4-EWSB-BlendedWind-GLO-025-6H-REPv6*.nc'))
        my = np.array([int(f[-76:-70]) for f in windfiles])
        windfiles = [windfiles[i] for i in np.where(np.logical_and(startyear*100+8 <= my, my <= (startyear+2)*100+9))[0]]
        winddimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
        windvariables = {'U': 'eastward_wind', 'V': 'northward_wind'}
        fset_wind = FieldSet.from_netcdf(windfiles, windvariables, winddimensions)
        fset_wind.add_periodic_halo(zonal=True, meridional=False, halosize=5)
        fset_wind.U.set_scaling_factor(withwind)
        fset_wind.V.set_scaling_factor(withwind)
        fname+='_wind%.4d' % (withwind*1000)

    fname += '.nc'

    if withstokes and withwind:
        fieldset = FieldSet(U=fset_currents.U+fset_stokes.U+fset_wind.U, V=fset_currents.V+fset_stokes.V+fset_wind.V)
    elif withstokes:
        fieldset = FieldSet(U=fset_currents.U+fset_stokes.U, V=fset_currents.V+fset_stokes.V)
    elif withwind:
        fieldset = FieldSet(U=fset_currents.U+fset_wind.U, V=fset_currents.V+fset_wind.V)
    else:
        fieldset = FieldSet(U=fset_currents.U, V=fset_currents.V)

    fieldset.add_field(Field('Kh_zonal', data=10*np.ones(size2D),
                                  lon=fset_currents.U.grid.lon, lat=fset_currents.U.grid.lat,
                                  mesh='spherical', allow_time_extrapolation=True))
    fieldset.add_field(Field('Kh_meridional', data=10*np.ones(size2D),
                                  lon=fset_currents.U.grid.lon, lat=fset_currents.U.grid.lat,
                                  mesh='spherical', allow_time_extrapolation=True))

    N = 1000

    pset = ParticleSet(fieldset=fieldset, pclass=JITParticle, lon=np.tile(-174.39, N),
                       lat=np.tile(-18.31, N), time=np.tile(datetime(startyear, 8, 7), N))

    pfile = ParticleFile(fname, pset, outputdt=delta(days=1))
    pset.execute(AdvectionRK4+pset.Kernel(BrownianMotion2D), dt=delta(hours=1), output_file=pfile, 
                 runtime=delta(days=730))
    pfile.close()
