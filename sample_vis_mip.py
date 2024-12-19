import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import os 
from PIL import Image

def img_show(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()
    
def show_nifi(img_path):
    img = nib.load(img_path).get_fdata()
    img_show(img, img_path.split('/')[-2])
    
def min_max_norm(arr:np.ndarray):
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    normalized_arr = (arr - arr_min) * (256 / (arr_max - arr_min))
    normalized_arr = normalized_arr.astype(np.uint16)
    return normalized_arr

    
def mip(src_img: np.ndarray, up:int = 255, down:int = 0):
    mip_img = np.ndarray((src_img.shape[0], src_img.shape[1]), dtype=float)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            mip_img[i, j] = np.max(src_img[i, j, down:up])
    img_show(mip_img)
    print('({},{})'.format(down, up))
    return mip_img

def mca_filter(src_img: np.ndarray, mca_mask: np.ndarray, val_used=None):
    if val_used is None:
        val_used = [2, 4, 6, 12, 14, 16]
        val_used = [2, 4, 12, 14]
    mca = np.zeros(src_img.shape).astype(float)
    for val in val_used:
        mca += np.where(mca_mask == val, 1, 0)
    # 返回mca区域的3D图像
    return src_img * mca

def mca_2d_mask(mca_mask: np.ndarray, val_used=None):
    if val_used is None:
        val_used = [2, 4, 6, 12, 14, 16]
        val_used = [2, 4, 12, 14]
    mca = np.zeros(src_img.shape).astype(float)
    for val in val_used:
        mca += np.where(mca_mask == val, 1, 0)
    # 返回mca区域的2D mask
    mca_2d = np.ndarray((mca_mask.shape[0], mca_mask.shape[1]), dtype=float)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            mca_2d[i, j] = np.max(mca[i, j, 0:255])
    return mca_2d

def mid_mip(src_img: np.ndarray, slice_num: int = 30):
    max_sum = 0
    max_idx = 0
    # 获取z轴最大脑区的index
    for i in range(src_img.shape[-1]):
        slice = src_img[:, :, i]
        slice_sum = np.count_nonzero(slice)
        if slice_sum > max_sum:
            max_sum = slice_sum
            max_idx = i
    # z轴上MIP的slice范围
    up = int(max_idx + 0.5 * slice_num)
    down = int(max_idx - 0.5 * slice_num)
    mip_img = np.ndarray((src_img.shape[0], src_img.shape[1]), dtype=float)
    for i in range(src_img.shape[0]):
        for j in range(src_img.shape[1]):
            mip_img[i, j] = np.max(src_img[i, j, down:up])
    img_show(mip_img)
    print('({},{})'.format(down, up))

    return mip_img

def vess_seg_jpg():
    root = 'D:\Datasets\PRoVe'
    src = 'mCTA1_brain.nii.gz'
    vess = 'mCTA1_vess.nii.gz'
    for id in os.listdir(root):
        if os.path.exists(os.path.join(root, id, vess)):
            src_img = nib.load(os.path.join(root, id, src)).get_fdata()
            src_img = src_img.transpose(0, 2, 1)
            vess_img = nib.load(os.path.join(root, id, vess)).get_fdata()
            vess_img = vess_img.transpose(0, 2, 1)
            img = src_img*vess_img
            vess_img = mip(vess_img)
            arr = src_img * vess_img
            arr = min_max_norm(arr)
            img = Image.fromarray(arr)
            img.save(os.path.join(root, id, 'mip_vess.jpg'), 'JPEG')

if __name__=='__main__':
    id_list = ['04-017','04-013','04-057','04-063', '01-152','04-077']
    path = 'D:/Datasets/MyRegistration/ProVe-IT-' + id_list[4]+'/lxy_reg_mCTA1.nii.gz'
    mca_path = 'D:/Datasets/MyRegistration/ProVe-IT-' + id_list[4]+'/lxy_reg_MCA.nii.gz'
    vess_path = 'D:\Datasets\PRoVe\ProVe-IT-01-001\mCTA1_vess.nii.gz'
    path2 = 'D:\Datasets\PRoVe\ProVe-IT-01-001\mCTA1_vess.nii.gz'
    
    src_img = nib.load(path).get_fdata()
    src_img = src_img.transpose(0, 2, 1)
    # res = mid_mip(src_img)
    mca_mask = nib.load(mca_path).get_fdata()
    mca_mask = mca_mask.transpose(0, 2, 1)
    mca_all = np.where(mca_mask > 0, 1, 0)
    src_img = mca_filter(src_img, mca_mask)
    # src_img = src_img*mca_all
    img = mid_mip(src_img=src_img)
    img_show(img, 'mid_mip')
    
    
    # src_img = mca_filter(src_img, mca_mask)
    mip_2d = mip(src_img, up=165,down=110)
    mca_2d = mca_2d_mask(mca_mask)
    # mca_img = mca_filter(src_img, mca_mask)
    # mca_mip_img = mid_mip(mca_img)
    img_show(mca_2d*mip_2d,'final')
    arr = mca_2d*mip_2d
    # arr = np.where(arr > 0, arr-50, 0)
    # arr = np.where(arr < 0, 0, arr)
    arr = min_max_norm(arr)
    img = Image.fromarray(arr)
    img = img.convert("L")
    img.save('output3.jpg','JPEG')
    # save res as jpg file
    '''
    img_pil = Image.fromarray(img_array)
            img_pil = img_pil.convert("L")
            img_pil.save(os.path.join(jpg_path, id + '.jpg'))
    '''
    
    
    