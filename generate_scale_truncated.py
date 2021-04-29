import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from skfmm import travel_time, distance
from scipy.interpolate import interp1d
import os
from utils import plot_2d_image
from math import log10

plt.style.use('ggplot')

def transform_normal_scores(scores, nscore):
    # for now, the values of our scores are less than the values of the nscore.
    # we dont have to worry about more extreme values. ultimately, we need some way to map an arbitrary distribution
    # function to these normal values.
    x = nscore['nscore']
    y = nscore['x']
    f = interp1d(x,y)
    return f(scores)

def linear_taper(n, inds=(0,-1), vals=(0.0,1.0) ):
    """
    Returns normalized coefficient for linear taper between (start, end) and 
    values (start_value, end_value)
    Args:
        n (int) : length of taper
        inds (tuple) : indexes of taper, default n
        vals (tuple) : coresponding to inds, default {0, 1.0}
    Returns:
        coef (ndarray) : coefficient {0 .. 1.0} of linear taper over indexes = inds with 
        values = vals
    """
    import numpy as np

    # vars
    ix = np.arange(n)
    coef = np.ones(n)

    # linear model
    delta_y = vals[1] - vals[0]
    if inds == (0,-1):
        delta_x = n
    else:
        delta_x = inds[1] - inds[0]
    slope = delta_y / delta_x
    intercept = vals[0] - slope * inds[0]
    coef[inds[0]:inds[1]] = slope * ix[inds[0]:inds[-1]] + intercept

    # returns
    return coef

def boundary_taper( field, taper_width=10, free_surface=True, values=0 ):
    """
    returns a field tapered along to boundary to zero. 
    can add taper to some percentage later.

    field (2d ndarray) : rupture field to taper.
    taper_width (int) : boundary to taper
    free_surface (bool) : (true) taper the free surface 
                          (false) do NOT taper free surface
    values sequence or int (optional) : ending values for taper. default is zero.  value should be specfied 
                                        in terms of percentages.

    return
    tapered_field (ndarray) : tapered field with shape = field.shape
    """
    ny, nx = field.shape
    if free_surface:
        baseline = np.ones( (ny-2*taper_width, nx-2*taper_width) )
        padded = np.pad( baseline, ((taper_width,taper_width), (taper_width,taper_width)), 'linear_ramp', end_values=values )
    else:
        baseline = np.ones( (ny-taper_width, nx-2*taper_width) )
        padded = np.pad( baseline, ((0,taper_width), (taper_width,taper_width)), 'linear_ramp', end_values=values )

    assert field.shape == padded.shape

    return field*padded


def main():
    plot_on = True
    tapering = True
    writing = True
    layered = False

    src_dir = './source_models/'
    output_name = 'sokrg-bbp_source1'
    out_dir = './source_models/source1'

    if not os.path.isdir( out_dir ):
        os.makedirs( out_dir )

    params = {
              'nx' : 273,
              'nz' : 136,
              'dx' : 100,
              'ihypo' : (120, 136),
              'fault_top' : 0,
              'avg_slip'  : 0.74,
              'std_slip'  : 0.42,
              'avg_psv'   : 1.42,
              'std_psv'   : 0.7,
              'avg_vrup'  : 0.79,
              'std_vrup'   : 0.04,
             }

    # read normal score transforms
    slip_sc = pd.read_csv('slip_nscore_transform_table.csv')
    psv_sc = pd.read_csv('psv_nscore_transform_table.csv')
    vrup_sc = pd.read_csv('vrup_nscore_transform_table.csv')

    # extract data
    slip_sim1 = np.fromfile(src_dir + 'slip_sim1.bin').reshape(params['nz'], params['nx'])
    psv_sim1 = np.fromfile(src_dir + 'psv_sim1.bin').reshape(params['nz'], params['nx'])
    vrup_sim1 = np.fromfile(src_dir + 'vrup_sim1.bin').reshape(params['nz'], params['nx'])

    if layered:
        material = np.loadtxt("bbp1d_1250_dx_25.asc")[params['fault_top']:params['fault_top'] + params['nz'], :]
        vp = material[:,1]*1e3
        vs = material[:,2]*1e3
        rho = material[:,3]*1e3

        vs = np.repeat(vs, params['nx']).reshape(params['nz'], params['nx'])
        rho = np.repeat(rho, params['nx']).reshape(params['nz'], params['nx'])

    else:
        vs = 3464*np.ones((params['nz'], params['nx']))
        rho = 2700*np.ones((params['nz'], params['nx']))

    # cut size of model down for computational ease
    slip = slip_sim1[:-1, :-1]
    psv = psv_sim1[:-1, :-1]
    
    # psv=(psv-psv.mean())/psv.std()
    vrup = vrup_sim1[:-1, :-1]
    vs = vs[:-1, :-1]
    rho = rho[:-1, :-1]

    # update parameters
    params['nx'] = params['nx'] - 1
    params['nz'] = params['nz'] - 1


    if tapering:
        # transform from normal-scores
        slip = transform_normal_scores(slip, slip_sc)
        psv = transform_normal_scores(psv, psv_sc)
        vrup = transform_normal_scores(vrup, vrup_sc)
        # according to xu et. al, 2016 for landers
        # taper = linear_taper( slip_sim1.shape[0], inds=(0, int(4000/params['dx'])), vals = (0.8, 1.0) )
        # slip = np.repeat(taper, params['nx']).reshape(params['nz'], params['nx']) * (slip_sim1 * params['std_slip'] + params['avg_slip'])

        # taper = linear_taper( psv_sim1.shape[0], inds=(0, int(4000/params['dx'])), vals = (0.5, 1.0) )
        # psv = np.repeat(taper, params['nx']).reshape(params['nz'], params['nx']) * (psv_sim1 * params['std_psv'] + params['avg_psv'])

        #vrup_norm = vrup_sim1 * params['std_vrup'] + params['avg_vrup']
        #taper = linear_taper( vrup_norm.shape[0], inds=(0, int(4000/params['dx'])), vals = (0.2, 1.0) )
        #vrup = np.repeat(taper, params['nx']).reshape(params['nz'], params['nx']) * vrup_norm * vs

        # from simulations, slip tapers larger
        taper_width = 37
        slip = boundary_taper(slip, 
                                taper_width=taper_width, 
                                free_surface=True, 
                                values=np.array(((0.60, 0.05), (0.05,0.05))) )
        
        # taper to 30% of mean along-strike psv at z = taper_width * dx 
        taper_width = 12
        ny,nx=psv.shape
        baseline = np.ones( (ny-4*taper_width, nx-2*taper_width) )
        padded = np.pad( baseline, 
                         ((3*taper_width,taper_width), (taper_width,taper_width)), 
                         'linear_ramp', 
                         end_values=np.array(((0.30, 0.05), (0.05,0.05))) )

        psv = padded * psv 
        vrup = vrup * vs

    # else:
    #     slip = slip_sim1 * params['std_slip'] + params['avg_slip']
    #     psv = psv_sim1 * params['std_psv'] + params['avg_psv']
    #     vrup_norm = vrup_sim1 * params['std_vrup'] + params['avg_vrup']
    #     vrup = vrup_norm * vs

    trup = compute_trup(vrup, params)
    
    if plot_on:
        # print(f'slip: min, max ({slip.min():.2f}, {slip.max():.2f})')
        # print(f'psv: min, max ({psv.min():.2f}, {psv.max():.2f})')
        # print(f'vrup: min, max ({vrup.min():.2f}, {vrup.max():.2f})')
        # print(f'trup: min, max ({trup.min():.2f}, {trup.max():.2f})')

        x = np.arange(0, params['nx'])
        z = np.arange(0, params['nz'])
        plotting_data = {'data':slip, 'contour':trup}
        plot_2d_image( plotting_data, "slip-" + output_name + ".pdf" , nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                              clabel = "Slip (m)", xlabel = "Distance (km)", ylabel = "Distance (km)",
                                              surface_plot = True, contour = True, clim=(0, slip.max()), cmap='jet' )
        plot_2d_image( psv, "psv-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                             clabel = r'$V^{peak}$ (m/s)', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                              surface_plot = False, contour = False, clim=(0, psv.max()), cmap='jet' )
        plot_2d_image( trup, "trup-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                              clabel = "Trup (s)", xlabel = "Distance (km)", ylabel = "Distance (km)",
                                              surface_plot = False, contour = True, clim=(0,20) )
        plot_2d_image( vrup/vs, "vrup-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                              clabel = r'$V_{rup}/c_s$', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                              surface_plot = False, contour = False, cmap='viridis', clim=(0, 1.0) )

        plot_2d_image( vs, "vs-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                              clabel = r'$V_{rup}/c_s$', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                              surface_plot = False, contour = False, cmap='jet' )

# generate strike, dip, and rake 
    nhat1 = np.fromfile("nhat1", "f").reshape(801,2601)
    nhat2 = np.absolute(np.fromfile("nhat2", "f").reshape(801,2601))
    nhat3 = np.fromfile("nhat3", "f").reshape(801,2601) # make vector point "up"
    
    # NOTE: starting at x=1000 to reduce model size for small model
    nhat1 = nhat1[::4, ::4]
    nhat2 = nhat2[::4, ::4]
    nhat3 = nhat3[::4, ::4]

    print(nhat1.shape)
    print(nhat2.shape)
    print(nhat3.shape)

#fienen "the three-point problem"
# project onto horizontal plane, calculate angle between
    dip = get_dip(nhat1, nhat2, nhat3)
    strike = get_strike(nhat1, nhat3)
    #rake = np.ones(strike.shape)*180.0 # constant rake
    rake = strike - 90 # strike is 270 and rake is 180
    
# compute moment 
    moment = get_moment(slip, vs, rho, params)
    print('moment')
    print(moment.sum())
    print(2./3 * (log10(moment.sum()) - 9.1))

# write to file for input
    if writing:
        dtype = '<f4'
        print('writing files...')
        # start at 1000 to reduce the size of the source simulation.
        vs.astype(dtype).tofile(os.path.join(out_dir, output_name + '_vs.bin'))
        rho.astype(dtype).tofile(os.path.join(out_dir, output_name + '_rho.bin'))
        slip.astype(dtype).tofile(os.path.join(out_dir, output_name + '_slip.bin'))
        psv.astype(dtype).tofile(os.path.join(out_dir, output_name + '_psv.bin'))
        vrup.astype(dtype).tofile(os.path.join(out_dir, output_name + '_vrup.bin'))
        trup.astype(dtype).tofile(os.path.join(out_dir, output_name + '_trup.bin'))
        strike.astype(dtype).tofile(os.path.join(out_dir, output_name + '_strike.bin'))
        dip.astype(dtype).tofile(os.path.join(out_dir, output_name + '_dip.bin'))
        rake.astype(dtype).tofile(os.path.join(out_dir, output_name + '_rake.bin'))
        moment.astype(dtype).tofile(os.path.join(out_dir, output_name + '_moment.bin'))
        

"""
Helping functions.
"""
def get_dip(nhat1, nhat2, nhat3):
    nz,nx = nhat1.shape
    dip = np.ones([nz,nx])

    for i in range(nz):
        for j in range(nx):
            nproj = (nhat1[i,j], 0, nhat3[i,j])
            n = (nhat1[i,j], nhat2[i,j], nhat3[i,j])
            norm = lambda v: np.sqrt(v[0]**2+v[1]**2+v[2]**2)
            scaling = 1.0 / ( norm(nproj) * norm(n) )
            arg = scaling*(n[0]**2+n[2]**2)
            if np.isclose(1.0, arg):
                arg = 1.0
            arg=np.arccos(arg)
            theta = np.rad2deg(arg)
            dip[i,j] = 90 - theta
    return dip

def get_moment(slip, vs, rho, params):
    mu = vs * vs * rho
    area = params['dx'] * params['dx']
    moment = mu * area * slip
    return moment


def get_strike(nhat1, nhat3):
    nz,nx = nhat1.shape
    strike = np.ones([nz,nx])
    for i in range(nz):
        for j in range(nx):
            nproj = (nhat1[i,j], 0, nhat3[i,j])
            x3 = (1,0,0)
            norm = lambda v: np.sqrt(v[0]**2+v[1]**2+v[2]**2)
            scaling = 1.0 / ( norm(x3) * norm( nproj) )
            theta = np.rad2deg(scaling * np.arccos(nproj[2]))
            if nhat1[i,j] > 0 and nhat3[i,j] > 0:
                strike[i,j] = 270 + theta
            elif nhat1[i,j] < 0 and nhat3[i,j] > 0:
                strike[i,j] = 270 - theta
            elif nhat1[i,j] < 0 and nhat3[i,j] < 0:
                # in 3rd quad
                strike[i,j] = 270 - theta
            elif nhat1[i,j] > 0 and nhat3[i,j] < 0:
                # in 4th quad
                strike[i,j] = theta - 90
    return strike

def source_time_function():
    pass

def compute_trup(vrup, params):
    phi = np.ones( (params['nz'], params['nx']) ) #* params['dx']
    print(phi.shape)
    ihypo = params['ihypo']
    phi[ ihypo[0], ihypo[1] ] = -1
    trup = travel_time( phi, speed=vrup, dx=params['dx'] )
    return np.array(trup)

if __name__ == "__main__":
    main()
