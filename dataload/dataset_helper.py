import scipy
import numpy as np
import torch


def get_padding_size(height, width, p=64):
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    # padding_left = (new_w - width) // 2
    padding_left = 0
    padding_right = new_w - width - padding_left
    # padding_top = (new_h - height) // 2
    padding_top = 0
    padding_bottom = new_h - height - padding_top
    return padding_left, padding_right, padding_top, padding_bottom

def ycbcr420_to_444(y, uv, order=1):
    '''
    y is 1xhxw Y float numpy array, in the range of [0, 1]
    uv is 2x(h/2)x(w/2) UV float numpy array, in the range of [0, 1]
    order: 0 nearest neighbor, 1: binear (default)
    return value is 3xhxw YCbCr float numpy array, in the range of [0, 1]
    '''
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def ycbcr444_to_420(yuv):
    '''
    input is 3xhxw YUV float numpy array, in the range of [0, 1]
    output is y: 1xhxw, uv: 2x(h/2)x(w/x), in the range of [0, 1]
    '''
    c, h, w = yuv.shape
    assert c == 3
    assert h % 2 == 0
    assert w % 2 == 0
    y, u, v = np.split(yuv, 3, axis=0)

    # to 420
    u = np.mean(np.reshape(u, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    v = np.mean(np.reshape(v, (1, h//2, 2, w//2, 2)), axis=(-1, -3))
    uv = np.concatenate((u, v), axis=0)

    y = np.clip(y, 0., 1.)
    uv = np.clip(uv, 0., 1.)

    return y, uv

def calc_psnr(img1, img2, data_range=255):
    '''
    img1 and img2 are arrays with same shape
    '''
    img1 = img1.astype(np.float16)
    img2 = img2.astype(np.float16)
    mse = np.mean(np.square(img1 - img2))
    if mse > 1e-10:
        psnr = 10 * np.log10(data_range * data_range / mse)
    else:
        psnr = 999.9
    return psnr


def get_curr_q_enc(q_scale, q_index=None):
    q_scale = np.exp(np.log(q_scale[-1] / q_scale[0])*q_index + np.log(q_scale[0]))
    return  q_scale     
def get_curr_q_dec( q_scale, q_index=None):
    q_scale = np.exp(np.log(q_scale[0])-np.log(q_scale[0] / q_scale[-1])*q_index )
    return q_scale  

