from parcels import FieldSet, ParticleSet, JITParticle, AdvectionRK4, ErrorCode, ParticleFile, Variable, Field, BrownianMotion2D, UnitConverter
from datetime import timedelta as delta
from datetime import datetime
import numpy as np
import math
import copy
from glob import glob

withstokes = True
withwind = 0.01  # scaling_factor

files = glob('global-analysis-forecast-phy-001-024_1567577968589.nc')
variables = {'U': 'uo', 'V': 'vo'}
dimensions = {'lon': 'longitude', 'lat': 'latitude', 'time': 'time'}
fset_currents = FieldSet.from_netcdf(files, variables, dimensions)
fset_currents.add_periodic_halo(zonal=True)
size2D = (fset_currents.U.grid.ydim, fset_currents.U.grid.xdim)

fname = 'pumiceevent_aug_2019'

if withstokes:
    stokesfiles = glob('global-analysis-forecast-wav-001-027_1567578677407.nc')
    stokesdimensions = {'lat': 'latitude', 'lon': 'longitude', 'time': 'time'}
    stokesvariables = {'U': 'VSDX', 'V': 'VSDY'}
    fset_stokes = FieldSet.from_netcdf(stokesfiles, stokesvariables, stokesdimensions)
    fset_stokes.add_periodic_halo(zonal=True)
    fname += '_wstokes'

if withwind:
    windfiles = sorted(glob('WIND_GLO_WIND_L4_NRT_OBSERVATIONS_012_004/2019/*.nc'))
    winddimensions = {'lat': 'lat', 'lon': 'lon', 'time': 'time'}
    windvariables = {'U': 'eastward_wind', 'V': 'northward_wind'}
    fset_wind = FieldSet.from_netcdf(windfiles, windvariables, winddimensions)
    fset_wind.add_periodic_halo(zonal=True)
    fset_wind.U.set_scaling_factor(withwind)
    fset_wind.V.set_scaling_factor(withwind)
    fname += '_wind%.4d' % (withwind * 1000)

fname += '.nc'

if withstokes and withwind:
    fieldset = FieldSet(U=fset_currents.U + fset_stokes.U + fset_wind.U,
                        V=fset_currents.V + fset_stokes.V + fset_wind.V)
elif withstokes:
    fieldset = FieldSet(U=fset_currents.U + fset_stokes.U, V=fset_currents.V + fset_stokes.V)
elif withwind:
    fieldset = FieldSet(U=fset_currents.U + fset_wind.U, V=fset_currents.V + fset_wind.V)
else:
    fieldset = FieldSet(U=fset_currents.U, V=fset_currents.V)

if withwind:  # for sampling
    for var, name in zip((fset_wind.U, fset_wind.V), ('uwind', 'vwind')):
        fld = copy.deepcopy(var)
        fld.name = name
        fld.units = UnitConverter()  # to sample in m/s
        fieldset.add_field(fld)

fieldset.add_field(Field('Kh_zonal', data=10 * np.ones(size2D),
                         lon=fset_currents.U.grid.lon, lat=fset_currents.U.grid.lat,
                         mesh='spherical', allow_time_extrapolation=True))
fieldset.add_field(Field('Kh_meridional', data=10 * np.ones(size2D),
                         lon=fset_currents.U.grid.lon, lat=fset_currents.U.grid.lat,
                         mesh='spherical', allow_time_extrapolation=True))

obslon = [-174.39, -174.68, -174.87, -174.96, -175.16, -175.32, -175.39, -175.47, -175.55, -175.6, -175.63, -175.68, -175.75, -175.82, -175.95, -176.11, -176.26, -176.3, -176.2, -176.51, -176.86, -177.3, -177.63, -178.01, -178.26, -178.58, -178.78]
obslat = [-18.31, -18.36, -18.31, -18.37, -18.44, -18.48, -18.55, -18.65, -18.67, -18.78, -18.69, -18.56, -18.46, -18.41, -18.43, -18.42, -18.53, -18.75, -18.55, -18.6, -18.61, -18.54, -18.46, -18.32, -18.21, -18.13, -18.17]

N = 100


def SampleWind(particle, fieldset, time):
    particle.uwind = fieldset.uwind[time, particle.depth, particle.lat, particle.lon]
    particle.vwind = fieldset.vwind[time, particle.depth, particle.lat, particle.lon]


class PumiceParticle(JITParticle):
    uwind = Variable('uwind', initial=fieldset.uwind)
    vwind = Variable('vwind', initial=fieldset.vwind)


pset = ParticleSet(fieldset=fieldset, pclass=PumiceParticle, lon=np.tile(obslon[0], N),
                   lat=np.tile(obslat[0], N), time=np.tile(datetime(2019, 8, 7), N))

pfile = ParticleFile(fname, pset, outputdt=delta(days=1))
pset.execute(AdvectionRK4 + pset.Kernel(BrownianMotion2D) + SampleWind, dt=delta(hours=1), output_file=pfile)
pfile.close()
