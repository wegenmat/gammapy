import numpy as np
from astropy.table import Table
from scipy import optimize, interpolate

# matplotlib inline
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib import rc
from matplotlib import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
rc('text', usetex=True)
rc('font', size=24)
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})


def spectrum(x, slope):
    y = x**slope

    return y


def timmerlc(slope, nt='None', dt='None', mean='None', sigma='None', seed='None'):
    # timmer alg from idl timmlc.pro
    if dt == 'None':
        dt = 1
    if nt == 'None':
        nt = 65536
    if mean == 'None':
        mean = 0
    if sigma == 'None':
        sigma = 1
    if seed == 'None':
        seed = 42

    simfreq = np.linspace(1, nt/2-1, num=nt/2, dtype='float64') / (dt*nt)
    simpsd = spectrum(simfreq, slope)
    fac = np.sqrt(simpsd)

    pos_real = np.random.RandomState(seed).normal(size=int(nt/2))*fac
    pos_imag = np.random.RandomState(seed).normal(size=int(nt/2))*fac

    pos_imag[int(nt/2)-1] = 0

    if float(nt/2.) > int(nt/2):
        neg_real = pos_real[0:int(nt/2)][::-1]
        neg_imag = -pos_real[0:int(nt/2)][::-1]
    else:
        neg_real = pos_real[0:int(nt/2)-1][::-1]
        neg_imag = -pos_real[0:int(nt/2)-1][::-1]

    real = np.hstack((0., pos_real, neg_real))
    imag = np.hstack((0., pos_imag, neg_imag))

    arg = real + 1j * imag
    rate = np.fft.ifft(arg).real
    time = dt*np.linspace(0, nt-1, nt, dtype='float')

    avg = np.mean(rate)
    std = np.sqrt(np.var(rate))

    rate = (rate - avg) * sigma / std + mean

    return time, rate


def periodogram(series):
    """computes the periodogram of the imput series"""
    series = series - np.mean(series)
    revff = np.fft.ifft(series)
    periodogram_bar = np.zeros(len(revff))
    freqs = np.zeros(len(revff))
    for freq, ff in enumerate(revff):
        if freq == 0:
            continue
        freqs[freq] = freq
        periodogram_bar[freq] = (abs(ff)**2)

    return freqs, periodogram_bar


def rebinlc(time, rate, dt):
    # check for finite values
    rate = rate[np.isfinite(time)]
    time = time[np.isfinite(time)]

    # rebin lc/psd to evenly spaced time binning, from rebinlc.pro
    t = time

    ts = t[0]
    num = int((t[-1] - ts) / dt + 1)

    minnum = 1

    tn = np.empty([num])
    rn = np.empty([num])
    numperbin = np.empty([num])

    tlimit = ts+dt*np.linspace(0, num, num+1)

    k = 0
    for i in range(num):
        tn[k] = tlimit[i]
        index = np.where(np.greater_equal(t, tlimit[i]) & (t < tlimit[i+1]))
        number = len(t[index])

        numperbin[k] = number
        if np.greater_equal(number, minnum):
            rn[k] = np.sum(rate[index]) / number
            k = k+1
        else:
            rn[k] = 0
            numperbin[k] = 0
            k = k+1

    if k == 0:
        print('Rebinned lightcurve would not contain any data')

    if k != num:
        tn = tn[0:k]
        rn = rn[0:k]
        numperbin = numperbin[0:k]

    tn = tn[numperbin != 0]
    rn = rn[numperbin != 0]

    return tn, rn


def do_periodogram(y):
    y = y - np.mean(y)
    freq, psd = periodogram(y)
    
    return freq, psd


def chi2_obs(norm, obs_pds, avg_pds, pds_err):
    # print(norm)
    obs_pds = norm * obs_pds
    a = (avg_pds - obs_pds)**2.
    b = pds_err**2.
    chi_obs = np.sum(a / b)
    # print(chi_obs)

    return chi_obs


def compare(obs_pds, avg_pds, pds_err, allpds, number_simulations):
    norm0 = 1.
    chi_obs = optimize.minimize(chi2_obs, norm0, args=(obs_pds, avg_pds, pds_err), method='SLSQP').fun
    
    chi_dist = np.empty([number_simulations])
    for n in range(number_simulations):
        a = (allpds[n, :] - avg_pds)**2
        b = pds_err**2
        chi = a / b
        chi_dist[n] = np.sum(chi)

    suf = 0
    for i in range(len(chi_dist)):
        if np.greater_equal(chi_dist[i], chi_obs):
            suf = suf + 1
    
    suf = suf / len(chi_dist)

    return suf


def psresp_pro(t, y, dy, slopes, number_simulations, binning, oversampling, df):
    bin = binning
    date, rat, raterr = t, y, dy
    date = date - date[0]
    duration = np.max(date) - np.min(date)
    npoints = int(duration / bin) * number_simulations * oversampling
    params = slopes
    lc_variance = np.var(rat) - np.var(raterr)
    # observed PDS calculation
    obs_nu, obs_pds = do_periodogram(rat)
    obs_nu = np.log10(obs_nu)
    # obs_nu = obs_nu[np.isfinite(obs_nu)]
    # obs_pds = obs_pds[np.isfinite(obs_nu)]

    # normalisation
    obs_pds = (2.*duration) / (np.mean(rat)*np.mean(rat)*len(rat)*len(rat)) * obs_pds

    # rebin
    # obs_freqs, obs_power = binlogPSD(obs_nu, obs_pds, df)
    obs_freqs, obs_power = rebinlc(obs_nu, obs_pds, dt=df)
    obs_power = np.log10(obs_power)    

    # create fake light curve
    faketime, fakerate = timmerlc(params, nt=npoints, dt=bin/oversampling)

    # calculate level of Poisson noise
    factor = ((len(raterr) / (2.*duration)) - (1. / duration))
    p_noise = np.sum(raterr**2.) / (len(raterr) * factor)

    # calculate high frequency aliased power
    uplim = 1. / (2.*bin)
    lowlim = 1. / (2.*(bin/10))
    intfreq = np.empty([int((lowlim-uplim)/uplim)+2])
    for i in range(len(intfreq)):
        intfreq[i] = uplim*(i+1)
    intpds = spectrum(intfreq, params)
    integral = np.trapz(intpds, x=intfreq)
    p_alias = integral / factor

    # long light curve is divided and resultant PDS are calculated
    allpds = np.empty([number_simulations, len(obs_freqs)])
    for j in range(number_simulations):
        # print('computing PDS ' + str(j+1))
        
        # indices for each segment
        lobin = int(duration * j / (bin/oversampling))
        hibin = int(duration * j / (bin/oversampling)) + int(duration/(bin/oversampling))

        # taken from appropriate section of light curve
        temptime = faketime[lobin:hibin]
        temprate = fakerate[lobin:hibin]

        # shift start time to zero
        temptime = temptime - temptime[0]

        # set bintime equal to original light curve time
        bintime = date
        binrate = np.interp(date, temptime, temprate)        
        
        # rescale simulated LC to the mean and variance of the original
        tempvar = np.sqrt(np.var(binrate))
        binrate = (binrate - np.mean(binrate)) * ((np.sqrt(lc_variance)) / tempvar) + np.mean(rat)

        # calculate PDS of simulated light curve
        sim_nu, sim_pds = do_periodogram(binrate)        
        sim_nu = np.log10(sim_nu)
        # sim_nu = sim_nu[np.isfinite(sim_nu)]
        # sim_pds = sim_pds[np.isfinite(sim_nu)]
        # sim_pds = sim_pds + p_noise + p_alias

        # normalisation
        sim_pds = (2.*(np.max(bintime)-np.min(bintime))) / (np.mean(binrate)*np.mean(binrate)*len(binrate)*len(binrate)) * sim_pds

        # rebin simulated PDS in same manner as observed
        # logfreqs, logpower = binlogPSD(sim_nu, sim_pds, df)
        logfreqs, power = rebinlc(sim_nu, sim_pds, dt=df)
        logpower = np.log10(power)

        # large array for later calculations of mean and rms error
        for k in range(len(logpower)):
            allpds[j, k] = logpower[k]
    
    avg_pds = np.empty([len(obs_freqs)])    
    pds_err = np.empty([len(avg_pds)])
    for i in range(len(avg_pds)):
        avg_pds[i] = np.mean(allpds[:, i])
    for i in range(len(pds_err)):
        pds_err[i] = np.sqrt(np.var(allpds[:, i]))

    return faketime, fakerate, obs_freqs, obs_power, bintime, binrate, avg_pds, pds_err, allpds


def psresp(t, dt, y, dy, slopes, number_simulations, binning, oversampling, df):
    # t, y = timmerlc(-1.6, nt=1000, dt=1)
    # t, y = t[300:700], y[300:700]
    # dt, dy = np.ones([len(t)]), np.ones([len(y)])
    t_ini, dt_ini, y_ini, dy_ini = t, dt, y, dy

    parameters = binning  # np.logspace(0, 2, 5)
    suf = np.empty([len(slopes), len(binning), len(df)])
    statistics = np.empty([4, len(binning), len(df)])
    for b in range(len(binning)):
        print('binning: ' + str(binning[b]))
        # t, dt, y, dy = binned(t_ini, dt_ini, y_ini, dy_ini, binning[b])
        t, y = rebinlc(t_ini, y_ini, binning[b])
        t, dy = rebinlc(t_ini, dy_ini, binning[b])
        # print(t, y)
        # all_avg_pds = []
        for s in range(len(slopes)):
            print('slope: ' + str(slopes[s]))
            for f in range(len(df)):
                print('df: ' + str(df[f]))

                # psresp
                faketime, fakerate, obs_freqs, obs_power, bintime, binrate, avg_pds, pds_err, allpds = psresp_pro(t, y, dy, slopes[s], number_simulations, binning[b], oversampling, df[f])
                # all_avg_pds.append(avg_pds)

                # do chi2
                suf[s, b, f] = compare(obs_power, avg_pds, pds_err, allpds, number_simulations)

                # find best slope and estimate error
                best_slope = slopes[np.argmax(suf[:, b, f])]
                best_slope_suf = np.max(suf[:, b, f])
                slopes_fwhw = interpolate.UnivariateSpline(-slopes, suf[:, b, f] - 0.5 * best_slope_suf, s=0).roots()
                # slopes_fwhw = slopes[np.less_equal(suf[:, b, f], 0.5 * best_slope_suf)]
                # low_slopes = slopes_fwhw[slopes_fwhw > best_slope]
                # high_slopes = slopes_fwhw[slopes_fwhw < best_slope]
                # print(slopes_fwhw)
                low_slopes = -slopes_fwhw[0]
                high_slopes = -slopes_fwhw[-1]
                # if (len(low_slopes) == 0) or (len(high_slopes) == 0):
                #     fwhw = np.nan
                # else:
                #     low = np.abs(np.min(low_slopes) - best_slope)
                #     high = np.abs(np.max(high_slopes) - best_slope)
                #     fwhw = low + high
                # if len(low_slopes) == 0:
                #     low_slopes = np.nan
                # if len(high_slopes) == 0:
                #     high_slopes = np.nan
                if low_slopes == high_slopes:
                    low_slopes = high_slopes = np.nan
                # low = np.abs(np.min(low_slopes) - best_slope)
                # high = np.abs(np.max(high_slopes) - best_slope)
                else:
                    low_diff = np.abs(low_slopes - best_slope)
                    high_diff = np.abs(high_slopes - best_slope)
                    # print(low, high)
            
                statistics[0, b, f] = best_slope
                statistics[1, b, f] = best_slope_suf
                # statistics[2, b, f] = fwhw
                statistics[2, b, f] = low_slopes
                statistics[3, b, f] = high_slopes
    
    bintime = bintime + t[0]

    # repo = str('/afs/ifh.de/group/amanda/scratch/wegenmat/ownCloud/Documents/data/statistical_analysis/' + source_id + '/SuF/')

    full_result = np.vstack((suf, statistics))
    # statistics_test = (statistics[1, :, :] > 0.95*np.max(suf)) & (len(statistics[2,:,:]) != 0) & (len(statistics[3,:,:]) != 0)
    statistics_test = (statistics[1, :, :] > 0.95*np.max(suf)) & (np.isfinite(statistics[2,:,:])) & (np.isfinite(statistics[3,:,:]))
    best_parameters = np.where(statistics_test == True)
    mean_slope = np.sum(statistics[0, :, :][statistics_test] * statistics[1, :, :][statistics_test]) / (np.sum(statistics_test))
    # mean_error = np.sqrt(np.sum((statistics[2, :, :][statistics_test] * statistics[1, :, :][statistics_test])**2))
    mean_error = np.abs(np.max(statistics[2, :, :][statistics_test]) - np.min(statistics[3, :, :][statistics_test]))

    data = Table()
    data['SuF'] = full_result
    # data.write(str(repo + 'SuF.fits'), overwrite=True)

    data = Table()
    data['statistics'] = statistics
    # data.write(str(repo + 'best_slopes.fits'), overwrite=True)

    data = Table()
    data['best_parameters'] = np.array([binning[best_parameters[0]], df[best_parameters[1]]])
    # data.write(str(repo + 'best_parameters.fits'), overwrite=True)

    data = Table()
    data['result'] = np.array([mean_slope, mean_error])
    # data.write(str(repo + 'result.fits'), overwrite=True)

    # prepare data for plot
    # data = np.empty([len(slopes)*len(binning)*len(df), 4])
    # i = 0
    # for s in range(len(slopes)):
    #     for b in range(len(binning)):
    #         for f in range(len(df)):
    #             data[i,:] = np.array([slopes[s],binning[b],df[f],SuF[s,b,f]])
    #             i = i+1

    # X, Y, Z, C = data[:,0], data[:,1], data[:,2], data[:,3]

    # name = str(repo + '+'.join(entry for entry in data_flag))
    fig = plt.figure(figsize=(11, 11))
    gs = gridspec.GridSpec(1, 1)
    # ax = fig.gca(projection='3d')
    # ax1 = fig.add_subplot(gs[0,:])
    # ax2 = fig.add_subplot(gs[1,:])
    ax3 = fig.add_subplot(gs[:, :])
    # ax4 = fig.add_subplot(gs[3,:])
    # ax5 = fig.add_subplot(gs[4,:])
    # ax1.plot(obs_freqs, obs_power)
    # ax1.set_xscale('log')
    # ax1.set_yscale('log')
    # for n in range(number_simulations):
    #     ax2.plot(obs_freqs, allpds[n,:])
    # for s in range(len(slopes)):
    #     ax2.plot(obs_freqs, all_avg_pds[s])
    # ax2.plot(freq_sim, pds_sim)
    # ax2.set_xscale('log')
    # ax2.set_yscale('log')
    # ax2.legend()
    for indx in range(len(best_parameters[0])):
        ax3.plot(-slopes, suf[:, binning == binning[best_parameters[0][indx]], df == df[best_parameters[1][indx]]], label='$ \Delta t_{{bin}} = {} $, $ \Delta f = {} $'.format(binning[best_parameters[0][indx]], df[best_parameters[1][indx]]))
    ax3.set(xlabel=r'\textbf{slope}',
            ylabel=r'\textbf{success fraction}',
            # , xlim=(0, max_period)
            ylim=(0, 1)
            )
    ax3.legend()
    # plt.show()
    plt.savefig('SuF', bbox_inches='tight')

    fig = plt.figure(figsize=(11, 11))
    ax = fig.gca(projection='3d')
    # sp = ax.scatter(X, Y, Z, s=100*C**4, c=C, alpha=.5)
    # plt.colorbar(sp)

    X, Y, Z = -slopes, binning, df

    suf_test = suf > 0.95*np.max(suf)
    sumz = np.sum(suf_test, axis=2)
    x, y = np.meshgrid(X, Y)
    xy = ax.contour(x, y, sumz.T, zdir='z', offset=0)

    sumy = np.sum(suf_test, axis=1)
    x, z = np.meshgrid(X, Z)
    xz = ax.contour(x, sumy.T, z, zdir='y', offset=7)

    sumx = np.sum(suf_test, axis=0)
    y, z = np.meshgrid(Y, Z)
    yz = ax.contour(sumx.T, y, z, zdir='x', offset=0.9)

    ax.set_xlim([0.9, 2.6])
    ax.set_ylim([1, 7])
    ax.set_zlim([0, 1.2])

    ax.set_xlabel(r'$slope$', labelpad=20)
    ax.set_ylabel(r'$\Delta t$', labelpad=20)
    ax.set_zlabel(r'$\Delta f$', labelpad=20)

    cxy = plt.colorbar(xy)
    cxy.ax.set_title(r'$slope$')
    cxz = plt.colorbar(xz)
    cxz.ax.set_title(r'$\Delta t$')
    cyz = plt.colorbar(yz)
    cyz.ax.set_title(r'$\Delta f$')

    # ax4.errorbar(t_ini, y_ini, dy_ini, fmt='ok')
    # ax4.plot(t, y, 'xk', alpha=0.5)
    # ax4.plot(t_ini, y_ini, 'or')
    # ax4.plot(t_test, y_test, 'ob', alpha=0.5)
    # for i in range(len(t)):
    #     ax4.hlines(y[i], t[i], t[i] + 6)
    # ax5.plot(bintime, binrate, 'ok')
    # for n in range(number_simulations):
    #     ax5.plot(t_sim[n,0,:], y_sim[n,0,:])
    # name = str(repo + '/' + '+'.join(entry for entry in data_flag))
    # plt.show()
    plt.savefig('Contour', bbox_inches='tight')

    # print('SuF ' + str(SuF))
    # print('best slopes ' + str(best_slopes))
    print('mean slope ' + str(mean_slope))
    print('mean error ' + str(mean_error))
    for indx in range(len(best_parameters[0])):
        print('used parameters: (binning ' + str(binning[best_parameters[0][indx]]) + ', df ' + str(df[best_parameters[1][indx]]) + ')')
    # print('saved ' + repo)

    return dict(mean_slope=mean_slope, mean_error=mean_error)
