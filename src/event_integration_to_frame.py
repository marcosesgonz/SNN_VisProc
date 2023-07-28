from spikingjelly import datasets as sjds
from typing import Callable, Dict, Optional, Tuple
import numpy as np
import os
import warnings
from spikingjelly.datasets import configure
np_savez = np.savez_compressed if configure.save_datasets_compressed else np.savez
warnings.filterwarnings("always", category=UserWarning, message='Detected a time step with no events within.')
warnings.filterwarnings("always", category=UserWarning, message='Detected a time step possibly corrupted. t_min != t[0]')

def exp_decay_func(amp,t,t_now,tau):
    """
    t: map of times or a given value of time
    t_0: actual time
    """
    outp = amp * np.exp((t-t_now)/tau)
    return outp

def mycal_fixed_frames_number_segment(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
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
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size
    
    if split_by == 'number':
        j_l = np.zeros(shape=[frames_num], dtype=int)
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N
        return j_l,j_r

    elif split_by == 'time':
        j_l = np.zeros(shape=[frames_num], dtype=int)
        dt = (events_t[-1] - events_t[0]) // frames_num
        #print(f't_0: {events_t[0]}, t_fin: {events_t[-1]}')
        idx = np.arange(N)
        for i in range(frames_num):
            t_l = dt * i + events_t[0]
            t_r = t_l + dt
            mask = np.logical_and(events_t >= t_l, events_t < t_r)
            #print(f' t_i:  {t_l}, t_fi: {t_r} ', np.all(mask ==False))
            idx_masked = idx[mask]
            if len(idx_masked) == 0:    #Este if es añadido por mi. 
                warnings.warn('Detected a time step with no events within.',UserWarning)
                j_l[i] = j_r[i-1]
                j_r[i] = j_l[i]
            else:
                j_l[i] = idx_masked[0]
                j_r[i] = idx_masked[-1] + 1
                if events_t[j_l[i]] != events_t[ j_l[i] : j_r[i]].min():
                    warnings.warn('Detected a time step possibly corrupted. t_min != t[0]',UserWarning)

        j_r[-1] = N  
        return j_l, j_r
    
    elif split_by == 'exp_decay':
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
    else:
        raise NotImplementedError

def leave_last_xyp_values_in_time(x,y,p):
    #np.unique puede devolver los índices de las primeras apariciones de los valores. Le doy la vuelta al array para obtener las posiciones de los últimos. Estas posiciones estarán 'invertidas'
    uniq_values, positions_inverted = np.unique(list(zip(x,y,p))[::-1],axis=0,return_index=True)
    #Revertimos posiciones
    positions = len(x) - 1 - positions_inverted
    return uniq_values,positions

def integrate_events_segment_to_frame_bydecay(t: np.ndarray, x: np.ndarray, y: np.ndarray, p: np.ndarray, H: int, W: int,
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
    if t.min() != t[0]:
        warnings.warn('Detected a time step possibly corrupted. t_min != t[0]',UserWarning)
    #Taking unique (x,y,p) values. Leaving only last values of (x,y,p) if repeated(keeping last in time).
    uniq_values, idx_positions = leave_last_xyp_values_in_time(x,y,p)
    #Asigning time values in x,y,p positions obtained
    x_uniq, y_uniq, p_uniq = uniq_values[:,0], uniq_values[:,1], uniq_values[:,2]
    frame[p_uniq, y_uniq, x_uniq] = t[idx_positions]
    #Calculating time decay
    frame_decayed = exp_decay_func(amp = scale_factor,t = frame, t_now = t[-1],tau = factor_tau*dt).astype(int)
    return frame_decayed


def myintegrate_events_by_fixed_frames_number(events: Dict, split_by: str, frames_num: int, H: int, W: int, factor_tau: float = 0.8, scale_factor: int = 50) -> np.ndarray:
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
    #Putting time from us to ms.(ASUMING TIME IN US(MICROSECONDS)) (Avoid memmory leakage)
    #t = t*1e-3
    if split_by == 'exp_decay':
        j_l_0, j_r, dt = mycal_fixed_frames_number_segment(events_t = t, split_by = split_by, frames_num = frames_num)
    else:
        j_l, j_r = mycal_fixed_frames_number_segment(events_t = t, split_by = split_by, frames_num = frames_num)

    frames = np.zeros([frames_num, 2, H, W]).astype('int')
    for i in range(frames_num):  
        if split_by == 'exp_decay':
            frames[i] = integrate_events_segment_to_frame_bydecay(t = t, x = x, y = y, p = p, H = H, W = W, j_l_0 = j_l_0, j_r = j_r[i],
                                                      dt = dt, factor_tau = factor_tau, scale_factor = scale_factor)
        else:
            frames[i] = sjds.integrate_events_segment_to_frame(x, y, p, H, W, j_l[i], j_r[i])
                
    return frames

#A continuation of integrate_events_file_to_frames_file_by_fixed_frames_number
def integrate_events_to_frame_wfixed_frames_num(loader: Callable, events_np_file: str, output_dir: str, split_by: str, frames_num: int, H: int, W: int,
                                                                print_save: bool = False, factor_tau: float = 0.8, scale_factor: int = 50) -> None:
    if events_np_file.endswith('.npy'):
        fname = os.path.join(output_dir, os.path.basename(events_np_file[:-4]))
    else:
        fname = os.path.join(output_dir, os.path.basename(events_np_file))
    np_savez(fname, frames=myintegrate_events_by_fixed_frames_number(loader(events_np_file),split_by, frames_num, H, W, factor_tau, scale_factor))
    if print_save:
        print(f'Frames [{fname}] saved.')
