# python imports
import os, shutil
from string import Template
from math import log10
import subprocess
import time

# global library imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# local imports
from krg_utils import *
from utils import plot_2d_image

def main(kwargs=None):

    print('Generating fault model using SO-KRG v1.0')
    print('========================================\n')

    params = kwargs or {}
    plot_on = params['plot_on']
    tapering = params['tapering']
    writing = params['writing']
    layered = params['layered']
    resample = False
    write_template = False
    force_slip_to_zero = True
    debug = False

    if params:
      for k, v in params.items():
        print(f'{k}: {v}')
    print('resample: ' + str(resample))
    print('write_template: ' + str(write_template))
    print('force_slip_to_zero: ' + str(force_slip_to_zero))
    print('debug: ' + str(debug))
    print()

    print('Generating random fields...')
    cmnd = [
        "RScript",
        "--vanilla",
        "generic_sim_tottori.R",
        str(params['output_path']),
        str(params['seed']),
        str(params['nsim']),
        str(params['dx']),
        str(params['fault_length']),
        str(params['fault_width'])
    ]
    return_code = subprocess.run(" ".join(cmnd), capture_output=True, shell=True)
    print(" ".join(cmnd))
    print(return_code.stdout.decode('utf-8'))

    if return_code.returncode != 0:
      print('\tError generating random fields. Exiting program.')
      print(return_code.stderr.decode('utf-8'))
      exit(1)

    # using array 1 index
    for src_idx in range(1, params['nsim']+1):

      print(f'Preparing source model {src_idx}...')

      src_dir = f'./source_models/'
      output_name = f'sokrg-bbp_source{src_idx}'
      out_dir = f'./source_models/source{src_idx}'

      if not os.path.isdir( out_dir ):
          os.makedirs( out_dir )

      # don't think we will use resampling to improve simultion times so hard-coding it out here
      params['nx'] = params['fault_length'] // params['dx'] + 1
      params['nz'] = params['fault_width'] // params['dx'] + 1
      params['inx'] = params['nx']
      params['inz'] = params['nz']

      # read normal score transforms, change to quantile transform
      slip_sc = pd.read_csv('slip_nscore_transform_table.csv')
      psv_sc = pd.read_csv('psv_nscore_transform_table.csv')
      vrup_sc = pd.read_csv('vrup_nscore_transform_table.csv')

      # extract data
      slip = np.fromfile(src_dir + f'slip_sim{src_idx}.bin').reshape(params['inz'], params['inx'])

      if force_slip_to_zero:
        slip = slip - slip.mean()

      if not debug:
        psv = np.fromfile(src_dir + f'psv_sim{src_idx}.bin').reshape(params['inz'], params['inx'])
        vrup = np.fromfile(src_dir + f'vrup_sim{src_idx}.bin').reshape(params['inz'], params['inx'])

        if resample:
            slip=resample2d(slip, shape=[params['nz'],params['nx']])
            psv=resample2d(psv, shape=[params['nz'],params['nx']])
            vrup=resample2d(vrup, shape=[params['nz'],params['nx']])
        else:
            params['nx'] = params['inx']
            params['nz'] = params['inz']
            # cut size of model down for computational ease
            slip = slip[:-1, :-1]
            psv = psv[:-1, :-1]
            # psv=(psv-psv.mean())/psv.std()
            vrup = vrup[:-1, :-1]
            # update parameters
            params['nx'] -= 1
            params['nz'] -= 1

        if layered:
            # bbp model storerd using kilometers
            material = expand_bbp_velocity_model(
              np.loadtxt(params['velocity_model_path']),
              params['nx'],
              params['nz'],
              params['dx'] * 1e-3
            )

            # convert to meters
            vp = material[0]*1e3
            vs = material[1]*1e3
            rho = material[2]*1e3

        else:
            vs = 3464*np.ones((params['nz'], params['nx']))
            rho = 2700*np.ones((params['nz'], params['nx']))

        # transform from normal-scores change this to normalized versions
        slip = transform_normal_scores(slip, slip_sc)
        psv = transform_normal_scores(psv, psv_sc)
        vrup = transform_normal_scores(vrup, vrup_sc)

        if tapering:
            avg_slip_pre = slip.mean()
            # from simulations, slip tapers larger
            taper_width = params['taper_width_slip']
            slip = boundary_taper(slip, 
                                    taper_width=taper_width, 
                                    free_surface=True, 
                                    values=np.array(((0.60, 0.05), (0.05,0.05))) )

            avg_slip_post = slip.mean()
            slip_taper_ratio = avg_slip_pre / avg_slip_post
            slip = slip * slip_taper_ratio


            # taper to 30% of mean along-strike psv at z = taper_width * dx 
            taper_width = params['taper_width_psv']
            ny,nx=psv.shape
            baseline = np.ones( (ny-4*taper_width, nx-2*taper_width) )
            padded = np.pad( baseline, 
                             ((3*taper_width,taper_width), (taper_width,taper_width)), 
                             'linear_ramp', 
                             end_values=np.array(((0.30, 0.05), (0.05,0.05))) )

            psv = padded * psv 
            vrup = vrup * vs
            
        else:
            vrup = vrup * vs

        # compute moment 
        print('Computing moment...')
        moment = get_moment(slip, vs, rho, params)
        moment_ratio = params['target_moment'] / moment.sum()
        if moment_ratio >= 1.1 or moment_ratio <= 0.9:
          print('Warning: greater than 10 percent different between simulated moment and target moment. Consider adjusting fault area.')
          print(f"\tTarget moment: {params['target_moment']}, Simulated moment: {moment.sum()}, Ratio: {moment_ratio}")

        # material model and fault area is constant; therefore, only change comes from slip
        slip = slip * moment_ratio
        moment = get_moment(slip, vs, rho, params)
        print(f'moment: {moment.sum()}\nmw: {2./3 * (log10(moment.sum()) - 9.05)}')
        print()
        trup = compute_trup(vrup, params)

    # compute new psv given tinti kinematic parameters

    # 1) cap max(slip/psv) = 2
        psv_eff=psv.copy()
        inds=np.where(slip/psv_eff > 2)
        psv_eff[inds]=slip[inds] / 2

    # 2) cap min(psv) = 0.1
        inds=np.where( psv_eff < 0.1 )
        psv_eff[inds]=0.1

    # estimate dcp based on mean of psv_eff and regression analysis, where vpeak/dcp = 2.46*fs_max
        fs_max = params['fs_max']
        ratio_vpeak_dcp = fs_max*2.46
        dc_est = 1.0/ratio_vpeak_dcp * psv_eff.mean()

    # compute ts on fault using dc_est
        ratio_dcp_est_psv_eff = dc_est / psv_eff
        ts = 1.55 * ratio_dcp_est_psv_eff

        # this was chosen as a reasonable upper bound, but this needs to be defined more explicitly
        # truptot=18.5
        # savran and olsen, 2020 defines this as the average trup on the fault boundary
        truptot = np.max([
          np.percentile(trup[0,:], 99),
          np.percentile(trup[-1,:], 99),
          np.percentile(trup[:,0], 99),
          np.percentile(trup[:,-1], 99)
        ])
        treff=truptot-trup

    # compute tr
        tr = 3.62 * slip + 0.07 * treff;
        tr[tr < 0] = 0

    # compute test tr using eq.11 from tinti et al, 2005
        tr_eq11 = (1.3*ts)/(dc_est/slip)**2
        tr_eq7 = slip**2 / psv_eff**2 / (1.3*ts)

    # compute psv of tinti functions
        inds=np.where(tr > 0)
        psv_tinti=np.zeros(psv_eff.shape)
        psv_tinti[inds] = 1.04*slip[inds] / ( ((1.3*ts[inds])**0.54) * (tr[inds]**0.47) )

        print(f'slip: min, max, mean ({slip.min():.2f}, {slip.max():.2f}, {slip.mean():.2f})')
        print(f'psv: min, max ({psv.min():.2f}, {psv.max():.2f})')
        print(f'psv/dcp: ({ratio_vpeak_dcp:.2f}) mean(psv_eff): ({psv_eff.mean():.2f}) dcp: ({dc_est:.2f})')
        print(f'vrup: min, max ({vrup.min():.2f}, {vrup.max():.2f})')
        print(f'trup: min, max ({trup.min():.2f}, {trup.max():.2f})')
        print(f'ts: min, max, mean ({ts.min():.2f}, {ts.max():.2f}, {ts.mean():.2f})')
        print(f'tr: min, max, mean ({tr.min():.2f}, {tr.max():.2f}, {tr.mean():.2f}')
        print(f'psv_tinti: min, max, mean ({psv_tinti.min():.2f}, {psv_tinti.max():.2f}, {psv_tinti.mean():.2f})')
        print(f'tr_eq11_eq11: min, max, mean ({tr_eq11.min():.2f}, {tr_eq11.max():.2f}, {tr_eq11.mean():.2f})')
        print(f'tr_eq7_eq11: min, max, mean ({tr_eq7.min():.2f}, {tr_eq7.max():.2f}, {tr_eq7.mean():.2f})')
        print(f'truptot: {truptot}')
        print()

        if plot_on:

            x = np.arange(0,params['nx'])
            z = np.arange(0,params['nz'])

            plotting_data = {'data': slip, 'contour': trup}
            plot_2d_image( plotting_data, out_dir + "/slip-" + output_name + ".pdf" , nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = "Slip (m)", xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = True, clim=(0, slip.max()), cmap='jet')

            plot_2d_image( psv, out_dir + "/psv-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                 clabel = r'$V^{peak}$ (m/s)', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, clim=(0, psv.max()), cmap='jet' )

            plot_2d_image( trup, out_dir + "/trup-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r"$t_{0}$ (s)", xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = True, clim=(0,12.5) )

            plot_2d_image( vrup/vs, out_dir + "/vrup-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r'$V_{rup}/c_s$', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, cmap='viridis', clim=(0, 1.0) )

            plot_2d_image( vs, out_dir + "/vs-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r'$c_s$', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, cmap='jet' )

            plot_2d_image( ts, out_dir + "/ts-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r'$\tau_s$ (s)', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, cmap='jet' )

            plot_2d_image( tr, out_dir + "/tr-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r'$\tau_r$ (s)', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, cmap='jet' )
            
            plot_2d_image( tr_eq11, out_dir + "/treq11-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r'$tr_{eq11}$', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, cmap='jet' )
            
            plot_2d_image( tr_eq7, out_dir + "/treq7-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r'$tr_{eq7}$', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, cmap='jet' )

            plot_2d_image( psv_tinti, out_dir + "/psv_tinti-" + output_name + ".pdf", nx = params['nx'], nz = params['nz'], dx = params['dx']*1e-3,
                                                  clabel = r'$V^{peak}$ (m/s)', xlabel = "Distance (km)", ylabel = "Distance (km)",
                                                  surface_plot = False, contour_plot = False, cmap='jet', clim=(0,psv.max()), show_plots=params['show_plots'])
            plt.close('all')
        # generate strike, dip, and rake 
        nhat1 = np.fromfile("nhat1", "f").reshape(801, 2601)
        nhat2 = np.absolute(np.fromfile("nhat2", "f").reshape(801, 2601))
        nhat3 = np.fromfile("nhat3", "f").reshape(801, 2601) # make vector point "up"
        
        # NOTE: starting at x=1000 to reduce model size for small model
        # should implement this calculation outside
        nhat1 = nhat1[::4]
        nhat2 = nhat2[::4]
        nhat3 = nhat3[::4]

        #fienen "the three-point problem"
        # project onto horizontal plane, calculate angle between
        print('Computing strike dip and rake...')
        dip = get_dip(nhat1, nhat2, nhat3)
        strike = get_strike(nhat1, nhat3)
        # rake = np.ones(strike.shape)*180.0 # constant rake
        rake = strike - 90 # strike is 270 and rake is 180

        # write to file for input
        if writing:
            dtype = '<f4'
            print('Writing files...')
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
            ts.astype(dtype).tofile(os.path.join(out_dir, output_name + '_ts.bin'))
            tr.astype(dtype).tofile(os.path.join(out_dir, output_name + '_tr.bin'))

            # making params.txt file
            if write_template:
              print('Writing parameter file to params.txt') 
              fin=open( 'params.tmpl' )
              template=Template( fin.read() )
              fin.close()
              d = {
               'psv_file': 'in/' + output_name + '_psv.bin',
               'vs_file': 'in/' + output_name + '_vs.bin',
               'rho_file': 'in/' + output_name + '_rho.bin',
               'trup_file': 'in/' + output_name + '_trup.bin',
               'strike_file': 'in/' + output_name + '_strike.bin',
               'dip_file': 'in/' + output_name + '_dip.bin',
               'rake_file': 'in/' + output_name + '_rake.bin',
               'slip_file': 'in/' + output_name + '_slip.bin',
               'momentrate_file': '../stripe_count_160/' + output_name + '_source.bin',
               'coord_file': 'in/fault_coords.bin',
               'dc': dc_est,
               'median_ts': 0.05333333,
               'truptot': truptot,
              }
              template=template.substitute(d)
              fout=open(os.path.join(out_dir,'params.txt'),'w')
              fout.write(template)
              fout.close()

              # copying fault_coords.bin 
              print('copying fault_coords.bin into out_dir')
              shutil.copy2('./fault_coords.bin', out_dir)

        

if __name__ == "__main__":
    params = {
      'fault_length': 27000,
      'fault_width': 14200,
      'dx' : 100,
      'target_moment': 8.62e+18,
      'ihypo' : (120, 135),
      'fault_top' : 0,
      'taper_width_slip': 30,
      'taper_width_psv': 10,
      'fs_max': 12.5,
      'output_path': '/Users/wsavran/Research/sokrg_bbp/source_models',
      'seed': 123456,
      'nsim': 1,
      'velocity_model_path': './central_japan_bbp1d.txt',
      'plot_on': True,
      'show_plots': True,
      'tapering': True,
      'writing': True,
      'layered': True
    }

    t0 = time.time()
    main(kwargs=params)
    t1 = time.time()
    print()
    print(f"Generated {params['nsim']} source models in {t1-t0} seconds.")