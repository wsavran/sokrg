import numpy as np
from scipy.interpolate import interp1d
from skfmm import travel_time, distance
from scipy.signal import resample

def resample2d( x, shape=[] ):
      if len(shape)==0:
          raise ValueError('shape should not be empty.')
      x1=resample(x,shape[0],axis=0)
      x2=resample(x1,shape[1],axis=1)
      return x2

def transform_normal_scores(scores, nscore):
    """
    map standard quantiles to empirical probability distribution from
    dynamic rupture simulation. values outside the empirical distribution
    are mapped to the ends. 
    """
    x = nscore['nscore']
    y = nscore['x']
    fill_value = (y.min(), y.max())
    f = interp1d(x,y,bounds_error=False,fill_value=fill_value)
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
    phi = np.ones( (params['nz'],params['nx']) ) #* params['dx']
    ihypo = params['ihypo']
    phi[ ihypo[0], ihypo[1] ] = -1
    trup = travel_time( phi, speed=vrup, dx=params['dx'] )
    return np.array(trup)


def expand_bbp_velocity_model(velocity_model_bbp_format, nx, nz, dx):
    """
    """

    # create array of discrete depths
    z = np.linspace(0, (nz-1)*dx, nz)

    # bbp provides layer thickness, so must convert to depth
    dep_inc = velocity_model_bbp_format[:,0]
    dep = np.cumsum(dep_inc)

    # look-up discrete depths in model
    layer_idxs = np.searchsorted(dep, z, side='right')

    # debugging stuff
    vs = np.zeros((nz, nx))
    vp = np.zeros((nz, nx))
    rho = np.zeros((nz, nx))
    
    for i, idx in enumerate(layer_idxs):
        # bbp format has cols: [layer_thickness, vp, vs, rho, qp, qs]
        vp[i,:] = velocity_model_bbp_format[idx, 1]
        vs[i,:] = velocity_model_bbp_format[idx, 2]
        rho[i,:] = velocity_model_bbp_format[idx, 3]

    return vp, vs, rho


if __name__ == "__main__":
    from utils import plot_2d_image
    mod = np.loadtxt('./central_japan_bbp1d.txt')
    nx = 273
    nz = 136
    dx = 0.1
    _, vs, _ = expand_bbp_velocity_model(mod, nx, nz, dx)

    ax = plot_2d_image(vs, nx=nx, nz=nz, dx=dx,
                       clabel = r'$c_s$ (km/s) ', xlabel="Distance (km)", ylabel="Distance (km)",
                       surface_plot=False, contour_plot=False)
    





