import pytest
import numpy as np
from numpy.testing import assert_allclose
from ..psresp import psresp


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

    return dict(t=time, y=rate)


TEST_CASES = [
    dict(slope=-1.6, nt=1000, dt=1, dy=np.ones([1000]),
         binning=np.array([1, 2, 3, 4]), oversampling=10, n_simulations=100,
         df=np.array([0.1, 0.2, 0.3, 0.4]), slopes=-np.array([1.5, 1.6, 1.7])
         )
]


@pytest.mark.parametrize('test_case', TEST_CASES)
def test_psresp(test_case):
    test_data = timmerlc(
        test_case['slope'], test_case['nt'], test_case['dt']
    )
    result = psresp(
        test_data['t'][300:700], test_data['dt'], test_data['y'][300:700], test_data['dy'][300:700],
        test_case['slopes'], test_case['n_simulations'],
        test_case['binning'], test_case['oversampling'], test_case['df'],
    )
    assert_allclose(result['mean_slope'], test_case['slope'], atol=result['mean_error'],)
