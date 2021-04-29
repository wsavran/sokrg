""" 
Imports
"""
import numpy as np
import scipy as sp
import scipy.signal as sig

"""
Self-explanatory helper functions
"""
def nextpow2(num):
    pow2 = 2
    while pow2 <= num:
        pow2 = pow2 * 2
    return pow2


"""
CALLABLE FUNCTIONS
"""
def vels2acc(time_series, dt):
    '''
    differentiates velocity to acceleration

    Parameters
    ----------
    time_series: numpy array
        time series

    dt: float
        time interval of time-series
    '''
    n = time_series.size
    acc = np.zeros([n,1])
    for i in range(1,n):
        acc[i] = (time_series[i]-time_series[i-1])/dt
    return np.squeeze(acc)

def vels2disp(time_series, dt):
    '''
        integrates velocity to displacement
    
        Parameters
        ----------
        time_series: numpy array
            time series
    
        dt: float
            time interval of time-series
    '''
    n = time_series.size
    disp = np.zeros([n,1])
    for i in range(1,n):
        disp[i] = disp[i-1]+dt*time_series[i]
    return disp 

def disp2vels(time_series, dt):
    '''
        differentiates displacement to velocity
    
        Parameters
        ----------
        time_series: numpy array
            time series
    
        dt: float
            time interval of time-series
    '''
    n = time_series.size
    vels = np.zeros([n,1])
    for i in range(1,n):
        vels[i] = (time_series[i]-time_series[i-1])/dt
    return vels

def acc2vels(time_series, dt):
    '''
        integrates acceleration to velocity
    
        Parameters
        ----------
        time_series: numpy array
            time series
    
        dt: float
            time interval of time-series
    '''
    n = time_series.size
    vels = np.zeros([n,1])
    for i in range(1,n):
        vels[i] = (time_series[i]-time_series[i-1])/dt
    return vels

def smooth_boxcar(data,degree):
    '''applies rectangular smoothing kernel to data.

    Parameters
    ----------
    data: numpy.array
        time-series to be smoothed
        
    degree: numpy.array
        order for smoothing kernel

    Returns
    -------
    smoothTriangle: numpy.array
        smoothed time series using rectangular kernel
    '''
    weights = np.ones(degree)*1/degree
    data = np.convolve(weights,data,mode='valid')
    return data
    
def smooth_triangle(data,degree,dropVals=False):
    '''applies triangular smoothing kernel to data.

    Parameters
    ----------
    data: numpy.array
        time-series to be smoothed
        
    degree: numpy.array
        order for smoothing kernel
        
    dropVals: float/numpy.array
        maintain original size and shape

    Returns
    -------
    smoothTriangle: numpy.array
        smoothed time series using triangular kernel
    '''
    triangle=np.array(range(degree)+[degree]+range(degree)[::-1])+1
    smoothed=[]
    for i in range(degree,len(data)-degree*2):
            point=data[i:i+len(triangle)]*triangle
            smoothed.append(sum(point)/sum(triangle))
    if dropVals: 
        return smoothed
    smoothed=[smoothed[0]]*(degree+degree/2)+smoothed
    while len(smoothed)<len(data):
        smoothed.append(smoothed[-1])
    return smoothed


def rotate_time_series(x, y, angle):
    '''Compute the rotated time series.

    Parameters
    ----------
    x: numpy.array
        first time series
        
    y: numpy.array
        second time series that is perpendicular to the first
        
    angle: float/numpy.array
        angle of rotation in degrees

    Returns
    -------
    rotated: numpy.array
        time series rotated by the specified angle
    '''

    angleRad = np.radians(angle)
    # Rotate the time series using a vector rotation
    return (x * np.cos(angleRad) + y * np.sin(angleRad), -y * np.sin(angleRad) + x * np.cos(angleRad))
    
def peak_ground_velocity(x, y, z):
    '''Compute the peak ground velocity.

    Parameters
    ----------
    x: numpy.array
        x component seismogram
        
    y: numpy.array
        y component seismogram

    Returns
    -------
    peak_ground_velocity: numpy.float64
        peak ground velocity at each component
    '''
    return (np.max(np.absolute(x)),
            np.max(np.absolute(y)),
            np.max(np.absolute(z)))    
            
def peak_ground_acceleration(x, y, z):
    '''Compute the peak ground acceleration.

    Parameters
    ----------
    x: numpy.array
        x component seismogram
        
    y: numpy.array
        y component seismogram
        
    z: numpy.array
        z component seismogram


    Returns
    -------
    peak_ground_velocity: numpy.float64
        peak ground acceleration
    '''
    return (np.max(np.absolute(x)),
            np.max(np.absolute(y)),
            np.max(np.absolute(z)))
    
def kinetic_energy_scalar(x, y, z, ttime, dt):
    '''Compute the kinetic energy scalar value for each component.

    Parameters
    ----------
    x: numpy.array
        x component seismogram
        
    y: numpy.array
        y component seismogram
        
    z: numpy.array
        z component seismogram
        
    ttime: float
        total time for integral. 
        
    dt: float
        delta time for the time-series

    Returns
    -------
    arias_intensity_scalar: numpy.array
        magnitude of the kinetic energy for the time series
    '''
    sim_ind = np.floor(ttime/dt)
    return (np.sum(x[:sim_ind]**2)*dt,
            np.sum(y[:sim_ind]**2)*dt,
            np.sum(z[:sim_ind]**2)*dt)  

def arias_intensity_scalar(x, y, z, ttime, dt):
    '''Compute the arias intensity value for each component.

    Parameters
    ----------
    x: numpy.array
        x component seismogram
        
    y: numpy.array
        y component seismogram
        
    z: numpy.array
        z component seismogram
        
    ttime: float
        total time for integral. 
        
    dt: float
        delta time for the time-series

    Returns
    -------
    arias_intensity_scalar: numpy.array
        magnitude of the kinetic energy for the time series
    '''
    sim_ind = np.floor(ttime/dt)
    const = np.pi/(2*981)
    return (const*np.sum(x[:sim_ind]**2)*dt,
            const*np.sum(y[:sim_ind]**2)*dt,
            const*np.sum(z[:sim_ind]**2)*dt)  
          
def energy_duration(x, y, z, t, ttime, dt):
    '''Computes energy duration for each component
    
    Parameters
    ----------
    x: numpy.array
        time series

    y: numpy.array
        time series     

    z: numpy.array
        time series     

    ttime: float
        total time for integral

    dt: float
        time interval of time-series
    
    Returns
    -------
    (f, out): (numpy.array, numpy.array)
        frequency vector for plotting
        fourier amplitude spectrum
    '''
    sim_ind = np.floor(ttime/dt)
    cke_x = np.cumsum(x**2)*dt
    cke_y = np.cumsum(y**2)*dt
    cke_z = np.cumsum(z**2)*dt
    norm_cke_x = cke_x / np.max(cke_x)
    norm_cke_y = cke_y / np.max(cke_y)
    norm_cke_z = cke_z / np.max(cke_z)
    y_start = 0.05
    y_end = 0.75
    t_start_x = np.interp(y_start, norm_cke_x, t)
    t_end_x = np.interp(y_end, norm_cke_x, t)
    t_start_y = np.interp(y_start, norm_cke_y, t)
    t_end_y = np.interp(y_end, norm_cke_y, t)
    t_start_z = np.interp(y_start, norm_cke_z, t)
    t_end_z = np.interp(y_end, norm_cke_z, t)
    dur_x = t_end_x - t_start_x
    dur_y = t_end_y - t_start_y
    dur_z = t_end_z - t_start_z
    return (dur_x, dur_y, dur_z)
      

def fourier_amplitude_spectrum(x, dt):
    '''Compute one-sided fourier amplitude spectrum
    
    Parameters
    ----------
    x: numpy.array
        time series
        
    dt: float
        time interval of time-series
    
    Returns
    -------
    (f, out): (numpy.array, numpy.array)
        frequency vector for plotting
        fourier amplitude spectrum
    '''
    # calculate length of fft
    nfft = nextpow2(len(x))
    
    # compute frequency vector
    f=np.fft.fftfreq(nfft, d=dt)
    
    # compute real fft
    x_fft = np.fft.fft(np.hanning(len(x))*x,n=nfft)
    
    # compute 1 sided amplitude spectrum
    x_fas = np.absolute(2*x_fft)
    
    # return values
    return(f[:nfft/2], x_fas[:nfft/2])

def envelope_function(x, y, z, ttime, dt):
    '''Compute envelope function based on hilbert transform

    Parameters
    ----------
    x: numpy.array
      time series

    y: numpy.array
      time series     

    z: numpy.array
      time series     

    ttime: float
      total time for integral

    dt: float
      time interval of time-series

    Returns
    -------
    (f, out): (numpy.array, numpy.array)
      frequency vector for plotting
      fourier amplitude spectrum
    '''
    sim_ind = np.floor(ttime/dt)
    analytic_x = np.absolute(sig.hilbert(x))
    analytic_y = np.absolute(sig.hilbert(y))
    analytic_z = np.absolute(sig.hilbert(z))
    return (np.sum(analytic_x[:sim_ind])*dt,
            np.sum(analytic_y[:sim_ind])*dt,
            np.sum(analytic_z[:sim_ind])*dt)


def lowpass(data, dt, fmax):
    '''lowpass filter a time series

    Parameters
    ----------
    data: numpy.array
      time series

    dt: float
      time spacing for time-series  

    fmax: float
      cutoff frequency for lowpass filter     

    Returns
    -------
    out: (numpy.array)
        lowpass filtered time-series
    '''
    ORDERLOW = 3
    RP = 0.5

    # filter data
    cornerlow = fmax / ((1./dt)/2)
    bl,al = sp.signal.cheby1(ORDERLOW, RP, cornerlow, 'low')
    data = sp.signal.filtfilt(bl,al,data)
    return data
            
    
    
def bandpass(data, dt, fc):
    '''bandpass filter a time series

    Parameters
    ----------
    data: numpy.array
      time series

    dt: float
      time spacing for time-series  

    [fmin, fmax]: float
      cutoff frequencies for bandpass filter     

    Returns
    -------
    out: (numpy.array)
        bandpass filtered time-series
    '''
    # filter constants
    ORDERLOW = 3        
    ORDERHIGH = 3
    RP = 0.5
    fmin = fc[0]
    fmax = fc[1]

    # filter data
    cornerlow = fmax / ((1./dt)/2)
    cornerhigh = fmin / ((1./dt)/2)
    bl,al = sp.signal.cheby1(ORDERHIGH, RP, cornerlow, 'low')
    temp = sp.signal.lfilter(bl,al,data)
    bh,ah = sp.signal.cheby1(ORDERHIGH, RP, cornerhigh, 'high')
    data = sp.signal.filtfilt(bh,ah,temp)
    return data

    
def highpass(data, dt, fmax):
    '''highpass filter a time series

    Parameters
    ----------
    data: numpy.array
      time series

    dt: float
      time spacing for time-series  

    fmax: float
      cutoff frequency for lowpass filter     

    Returns
    -------
    out: (numpy.array)
        highpass filtered time-series
    ''' 
    
    ORDERHIGH = 3
    RP = 0.5

    # filter data
    cornerhigh = fmax / ((1./dt)/2)
    bh,ah = sp.signal.cheby1(ORDERHIGH, RP, cornerhigh, 'high')
    data = sp.signal.filtfilt(bh,ah,data)
    return data

def compute_spectral_acceleration(accx, accy, dtt, T, c=0.05, NintMax=10): 
    '''
    rotdsa.py
    author: william savran
    date: 1.27.2015
    
    computes SA_rotdN given two orthogonal time-series using a time-domain ODE solver. originally written
    by S.M. Day and ported to python by william savran
    
    Parameters
    ----------
    accx: numpy.array
        acceleration time-series
    
    accy: numpy.array
        acceleration time-series
    
    dtt: float
        time-step of acceleration time-series
        
    T: numpy.array, float
        set of periods to calculate SA at
    
    c: float
        damping coefficient. default 5%
        
    NintMax: int
        interpolation parameter. default = 10

    '''
    # convert periods to numpy arrays
    T = np.array(T)
    
    # Do some error handling
    assert NintMax < 100, ' NintMax cannot exceed 100'
    minT = 4*dtt/NintMax
    assert np.min(T) > minT, 'Minimum period must be greater than %f' % minT


    # prepare variables, such as eigenvalues and time-integration parameters
    Nint = np.ceil(4*dtt/np.min(T))
    # eigenvalues
    l1 = np.complex(-c, np.sqrt(1-c**2))
    l2 = np.complex(-c, -np.sqrt(1-c**2))
    
    C = np.array([[l1,l2],[1,1]])
    R = np.array((1/(2j)/np.sqrt(1-c**2))*np.array([[1,-l2],[-1,l1]]))
    
    # time step for integrator
    dt = dtt/Nint
    om0 = 2*np.pi/T
    
    # E matrix
    E11 = np.array(np.exp(l1*dt*om0))
    E22 = np.array(np.exp(l2*dt*om0))
    Eint11 = (1./l1/om0)*(1-E11)
    Eint22 = (1./l2/om0)*(1-E22)
    
    # loop through each component
    for component in range(2):
        sample=np.ceil(0.5*max(T)/dtt)
        if (component==0):
            acc=accx
        else:
            acc=accy
            
        Unew=np.zeros((2,len(om0)),dtype='c8')
        Uold=np.zeros((2,len(om0)),dtype='c8')
        
        aEx=np.append(acc, np.zeros(sample));
        aExInterp = []
        for j in range(len(aEx)-1):
            for k in range(int(Nint)):
                aExInterp.append(aEx[j]*(Nint-k)/Nint+aEx[j+1]*(k)/Nint)
                
        aEx=np.array(aExInterp)
        Ntot=len(aExInterp)
        if (component == 0):
            ux = np.zeros((Ntot,len(T)),dtype='c8')
        else:
            uy = np.zeros((Ntot,len(T)),dtype='c8')
    
        # Form propagator C*E*R and Integrate the ODE
        for j in range(Ntot):
            Unew[0,:]=R[0,0]*(aEx[j]*Eint11+Uold[0,:]*E11)+R[0,1]*Uold[1,:]*E11
            Unew[1,:]=R[1,0]*(aEx[j]*Eint22+Uold[0,:]*E22)+R[1,1]*Uold[1,:]*E22
            Unew=np.array(np.matrix(C)*np.matrix(Unew))
            Uold=np.copy(Unew)
            if (component == 0):
                ux[j,:]=Unew[1,:]
            else:
                uy[j,:]=Unew[1,:]

    GeoMax = np.zeros((90,len(T)))
    for j in range(90):
        cth = np.cos(j*np.pi/180)
        sth = np.sin(j*np.pi/180)
        osc1 = cth*ux+sth*uy
        osc2 = -sth*ux+cth*uy
        GeoMax[j,:]=np.sqrt(np.amax(np.absolute(osc2),axis=0)*np.amax(np.absolute(osc1),axis=0))
    # this can be needs modified to get other angles.
    Sa = om0*np.median(GeoMax, axis=0)
    return Sa

'''
    compute back azimuth given a source/receiver location

    Parameters
    ----------
    x: array like
        x coordinates of receiver locations
    
    y: array like  
        y coordinates of receiver locations
    
    source: tuple
        (x,y) locations of the source

'''
def get_back_azimuth(x,y,source):
    x_off = x - source[0]
    y_off = y - source[1]
    az = np.arctan(x_off/y_off)
    if x_off > 0:
        ba = 180. + az
        return ba
    elif x_off < 0:
        ba = 180. - az
        return ba
    elif x_off == 0.0:
        ba = 180.
        return ba
    

'''
    rotate time series to transverse and radial components
    
    Parameters
    ----------
    xt: numpy.array
        x time-series
    
    yt: numpy.array
        y time-series
    
    ba: float
        back-azimuth angle
'''
# might not work
def rotate_back_azimuth(xt,yt,ba):
    r = xt * np.sin((ba + 180.) * 2 * np.pi / 360.) + yt * np.cos((ba + 180.) * 2 * np.pi / 360.)
    t = xt * np.cos((ba + 180.) * 2 * np.pi / 360.) - yt * np.sin((ba + 180.) * 2 * np.pi / 360.)
    return (r,t)

"""
Code ported to python from Bo Jacobsen @ Aarhus University by William Savran
generates realization of von karman process in 1d using autocorrelations.
"""
def selfsimilar(rs, nu, L):
    rho = np.spacing(1) + np.abs(rs)/L;
    phi = rho**nu*kv(nu, rho)
    return phi


def realize_von_karman(rs, nu, L):
    N = rs.size
    C_0 = 0.05 #
    C_a = np.zeros([N,N])
    for i_row in xrange(N):
        C_a[i_row,:] = C_0*selfsimilar(rs-rs[i_row], nu, L)
    C_a = C_a+C_0*1e-12*np.eye(np.max(C_a.shape))
    w,v = eig(C_a)
    sqrtC_a = np.matrix(v)*np.matrix(np.diag(np.sqrt(w)))*np.matrix(v).T
    d_Nsim = np.matrix(sqrtC_a) * np.matrix(randn(N,1))
    return d_Nsim


def get_backends():
    """ copied from pelson's comment @ http://stackoverflow.com/questions/5091993/list-of-all-available-matplotlib-backends """
    import matplotlib.backends
    import os.path

    def is_backend_module(fname):
        """Identifies if a filename is a matplotlib backend module"""
        return fname.startswith('backend_') and fname.endswith('.py')

    def backend_fname_formatter(fname): 
        """Removes the extension of the given filename, then takes away the leading 'backend_'."""
        return os.path.splitext(fname)[0][8:]

    # get the directory where the backends live
    backends_dir = os.path.dirname(matplotlib.backends.__file__)

    # filter all files in that directory to identify all files which provide a backend
    backend_fnames = filter(is_backend_module, os.listdir(backends_dir))
    backends = [backend_fname_formatter(fname) for fname in backend_fnames]

    print(backends)


def plot_2d_image( input, filename=None, nx=None, nz=None, dx=1.0, clabel=None, xlabel=None, 
                   ylabel=None, surface_plot=False, contour_plot=False, cmap='viridis', show_plots=False, **kwargs ):
    """Plots 2d array with modified colorbar and extra options.

    Args:
        input (ndarray)      : (list) 2d array to be plotted, (dict) if contour is True dict will contain 
                               the 2d array that will be used for contouring under the key 'contour'
        nx (int)             : number of nodes in x direction
        nz (int)             : number of nodes in z direction
        dx (float)           : grid spacing
        label (str)           : units of array, e.g. if array contains velocities units would be 'Velocity (m/s)'
        surface_plot (bool)  : plot axis above 2d image plot showing the surface trace of array
        contour (bool)       : add contour to 2d image, if input is dict plot input['contour'] as the contour
                               else plot the contour of input
        **kwargs (dict)      : any args to be passed on
    """
    import matplotlib
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from matplotlib.pyplot import colorbar, figure, show
    import os

    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    fig = figure()
    ax = fig.add_subplot(111)

    # fast x convention
    ex = nx * dx
    ez = nz * dx

    # handle different input types
    if isinstance(input, np.ndarray):
        data = input
        contour_self = True
    else:
        data = input['data']
        contour_data = input['contour']
        contour_self = False

    # plot axis normal
    im = ax.imshow(data, extent=(0, ex, 0, ez), origin='lower', cmap=cmap)

    # contour
    if contour_plot:
        # print('contouring...')
        x = np.arange(0,ex,dx)
        z = np.arange(0,ez,dx)
        xx, zz = np.meshgrid(x,z)
        v = 1.0 * np.arange(-25,25)
        if contour_self:
            ctrup = ax.contour( xx, zz, data, v, 
                            extent=(0, ex, 0, ez), colors='gray', 
                            linewidths=0.25, antialiased=False )
        else:
            ctrup = ax.contour( xx, zz, contour_data, v, 
                            extent=(0, ex, 0, ez), colors='gray', 
                            linewidths=0.25, antialiased=False )

    divider = make_axes_locatable(ax)

    if surface_plot:
        x = np.arange(0,ex,dx)
        tax = divider.append_axes("top", size="25%", pad=0.05)
        tax.plot(x, data[0,:], 'k')
        tax.set_yticks([0, data[0,:].max()])
        tax.tick_params(
                axis = 'x',
                which = 'both',
                bottom = 'off',
                right = 'off',
                labelbottom = 'off',
                )
        ticks = tax.yaxis.get_majorticklabels()
        ticks[0].set_verticalalignment('bottom')
        ticks = ax.yaxis.get_majorticklabels()
        ticks[0].set_verticalalignment('top')
        ax.tick_params(axis='x', top = 'off', labeltop = 'off')


    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = colorbar(im, cax=cax)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    cbar.set_label(label=clabel, size=14)
    ax.set_ylim([ez,0])
    ax.set_xlim([0,ex])
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    if 'clim' in kwargs:
        im.set_clim(kwargs['clim'])
    else:
        im.set_clim([data.min(),data[np.where(data < 1e9)].max()]) # hacky

    if filename:
        fig.savefig( filename, dpi=300 )

    if show_plots:
        show()

    return ax

def compute_rupture_velocity(trup, dx):
    import os
    import numpy as np
    import numpy.matlib as ml

    
    # if isinstance(cs, np.ndarray):
    #     print 'repeating...'
    #     cs = ml.repmat(cs[:-1],nx,1).T
    # trup_ma = np.ma.masked_values( trup, 1e9 )
    trup_ma = np.ma.masked_values(trup,1e9)
    gy, gx = np.absolute(np.gradient(trup_ma))
    ttime = np.sqrt(gy**2 + gx**2)
    vrup = dx / ttime
    return vrup


def parse_simulation_details( cwd, write = False ):
    import os
    import logging
    # data structure for simulation.
    data = {}
    data['parameters'] = {}
    data['fieldio'] = {}
    # read meta.py file
    try:
        # this is dangerous, should change to json 
        exec( open( os.path.join(cwd, 'meta.py')).read() )
        # get list of local variables aka namespace of meta.py
        lvars = locals()
        exclude = ['json', 'lvars', 'shape', 'xi', 'indices']
        for var, val in lvars.items():
            # exclude builtin types and json import
            if not var.startswith('__') and var not in exclude:
                if var == 'fieldio':
                    inputs, outputs = _parse_fieldio(val, eval('shape'), eval('indices'))
                    data['fieldio']['inputs'] = inputs
                    data['fieldio']['outputs'] = outputs
                else:
                    data['parameters'][var] = eval(var)

        
        # write json file containing simulation data
        if write:
            import json
            with open('test2.js', 'w') as fh:
                json.dump(data, fh, indent=2)
    
    except Exception as e:
        logging.error('cannot read simulation details. error: %s' % str(e)) 
        return data
    
    return data

"""turns meta.py file into json object using eval, this is very risky, but I trust myself"""
def _parse_fieldio(fieldio, shape, indices):
    inputs = []
    outputs = []
    for field in fieldio:
        # outputs
        field_vals = field[-3:]
        if field[0] == '=w':
            outputs.append( {
                'file': field_vals[0],
                'field': field_vals[2][0],
                'shape': shape[str(field_vals[0])],
                'indices': indices[str(field_vals[0])],
                } )

        # inputs
        if field[0] == '=R':
            if field_vals[0] == '-':
                inputs.append( { 
                    'file': '',
                    'field': field_vals[2][0],
                    'val': field_vals[1],
                 } )
            else:
                outputs.append( {
                    'file': field_vals[0],
                    'field': field_vals[2][0],
                    'val': ''
                    } )
        
    return inputs, outputs

def nicecolorbar(ax, clabel=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = colorbar(im, cax=cax)
    cbar.solids.set_rasterized(True)
    cbar.solids.set_edgecolor("face")
    cbar.set_label(label=clabel, size=14)
