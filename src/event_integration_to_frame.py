from spikingjelly import datasets as sjds
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import os
from spikingjelly.datasets import configure
np_savez = np.savez_compressed if configure.save_datasets_compressed else np.savez

def exp_decay_func(amp,t,t_now,tau):
    """
    t: map of times or a given value of time
    t_0: actual time
    """
    outp = amp * np.exp((t-t_now)/tau)
    return outp

def cal_fixed_frames_number_segment_index_by_time(events_t: np.ndarray, frames_num: int) -> tuple:
    '''
    :param events_t: events' t
    :type events_t: numpy.ndarray
    :param split_by: 'time' or 'number'
    :type split_by: str
    :param frames_num: the number of frames
    :type frames_num: int
    :return: a tuple ``(j_l, j_r, dt)``  #Devuelve dos arrays con las posiciones de los índices del array tiempo donde empieza(j_l) y acaba(j_r) cada frame. Además devuelve el paso temporal en unidades de tiempo
    :rtype: tuple
    Denote ``frames_num`` as :math:`M`
    .. math::(split by time)
        \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
        j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
        j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}
    '''
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size

    dt = (events_t[-1] - events_t[0]) // frames_num
    idx = np.arange(N)
    for i in range(frames_num):
        #t_l = dt * i + events_t[0]
        t_r = events_t[0] + dt*(i +1)
        mask = np.logical_and(events_t >= events_t[0], events_t < t_r)
        idx_masked = idx[mask]
        if i == 0:
            j_l_0 = idx_masked[0]
        j_r[i] = idx_masked[-1] + 1

    j_r[-1] = N
    return j_l_0, j_r, dt

def leave_last_xyp_values_in_time(x,y,p):
    #np.unique puede devolver los índices de las primeras apariciones de los valores. Le doy la vuelta al array para obtener las posiciones de los últimos. Estas posiciones estarán 'invertidas'
    uniq_values, positions_inverted = np.unique(list(zip(x,y,p))[::-1],axis=0,return_index=True)
    #Revertimos posiciones
    positions = len(x) - 1 - positions_inverted
    return uniq_values,positions

def integrate_events_segment_to_frame(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int,
                                       j_l_0: int = 0, j_r: int = -1,dt:float = 1e6,factor_tau: float = 1.5, scale_factor: int = 50) -> np.ndarray:
    '''
    :param x: x-coordinate of events
    :type x: numpy.ndarray
    :param y: y-coordinate of events
    :type y: numpy.ndarray
    :param p: polarity of events
    :type p: numpy.ndarray
    :param H: height of the frame
    :type H: int
    :param W: weight of the frame
    :type W: int
    :param j_l: the start index of the integral interval, which is included
    :type j_l: int
    :param j_r: the right index of the integral interval, which is not included
    :type j_r:
    :return: frames
    :rtype: np.ndarray
    Denote a two channels frame as :math:`F` and a pixel at :math:`(p, x, y)` as :math:`F(p, x, y)`, the pixel value is integrated from the events data whose indices are in :math:`[j_{l}, j_{r})`:
    '''

    frame = np.zeros([2, H, W])
    t = t[j_l_0:j_r]
    x = x[j_l_0:j_r]
    y = y[j_l_0:j_r]
    p = p[j_l_0:j_r]

    #Taking unique (x,y,p) values. Leaving only last values of (x,y,p) if repeated(keeping last in time).
    uniq_values, idx_positions = leave_last_xyp_values_in_time(x,y,p)
    #Asigning time values in x,y,p positions obtained
    x_uniq, y_uniq, p_uniq = uniq_values[:,0], uniq_values[:,1], uniq_values[:,2]
    frame[p_uniq, y_uniq, x_uniq] = t[idx_positions]
    #Calculating time decay
    frame_decayed = exp_decay_func(amp = scale_factor,t = frame, t_now = t[-1],tau = factor_tau*dt).astype(int)
    return frame_decayed


def integrate_events_by_fixed_frames_number_bydecay(events: Dict, frames_num: int, H: int, W: int, factor_tau: float = 1.5, scale_factor: int = 50) -> np.ndarray:
    '''
    :param events: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
    :type events: Dict
    :param frames_num: the number of frames
    :type frames_num: int
    :param H: the height of frame
    :type H: int
    :param W: the weight of frame
    :type W: int
    :return: frames
    :rtype: np.ndarray
    Integrate events to frames by fixed frames number. 
    '''
    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
    #Putting time from us to s.(ASUMING TIME IN US(MICROSECONDS))
    #t *= 1e-6
    j_l_0, j_r, dt = cal_fixed_frames_number_segment_index_by_time(t, frames_num)
    frames = np.zeros([frames_num, 2, H, W]).astype('int')
    for i in range(frames_num):
       frames[i] = integrate_events_segment_to_frame(t = t, x = x, y = y, p = p, H = H, W = W, j_l_0 = j_l_0, j_r = j_r[i],
                                                      dt = dt, factor_tau = factor_tau, scale_factor = scale_factor)
    return frames


def exp_decay_by_fixed_time(loader: Callable, events_np_file: str, output_dir: str, frames_num: int, H: int, W: int,
                             print_save: bool = False,factor_tau:float = 1.5, scale_factor:int = 50) -> None:
    fname = os.path.join(output_dir, os.path.basename(events_np_file))
    np_savez(fname, frames=integrate_events_by_fixed_frames_number_bydecay(loader(events_np_file), frames_num, H, W, factor_tau, scale_factor))
    if print_save:
        print(f'Frames [{fname}] saved.')
    return None