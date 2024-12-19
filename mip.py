import os

import nibabel as nib
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def img_show(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def show_nifi(img_path):
    img = nib.load(img_path).get_fdata()
    img_show(img, img_path.split('/')[-2])


# 固定slice数量的mip操作
def basic_mip(src_img: np.ndarray, slice_num: int = 30):
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


def ratio_mip(src_img: np.ndarray, ratio: float = 0.25):
    b_len = 0  # 脑区长度
    max_idx = 0  # 最大脑区面积切片坐标
    max_sum = 0  # 最大脑区面积
    for i in range(src_img.shape[-1]):
        slice = src_img[:, :, i]
        cu_sum = np.count_nonzero(slice)
        if cu_sum > 0:
            b_len += 1
        if cu_sum > max_sum:
            max_sum = cu_sum
            max_idx = i
    # ratio = 0.15
    up = int(max_idx + b_len * ratio * 0.5)
    down = int(max_idx - b_len * ratio * 0.5)
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


def vess_filter(src_img: np.ndarray, vess_mask: np.ndarray):
    vess = np.zeros(src_img.shape).astype(float)
    vess = np.where(vess_mask != 0, 1, 0)
    # 返回血管分割mask过滤的结果
    return src_img * vess


def mca_ratio_mip(src_img: np.ndarray, mca_mask: np.ndarray, ratio: float = 0.25):
    # 获取MCA区域脑组织
    mca_img = mca_filter(src_img, mca_mask)
    # 对获取后MCA区域的图像进行MIP操作
    mip_img = ratio_mip(mca_img, ratio)
    # print('1')
    return mip_img


def vess_40_mip(src_img: np.ndarray, vess_mask: np.ndarray, slice_num: int = 40):
    # 获取40%血管区域脑组织
    vess_img = vess_filter(src_img, vess_mask)
    # 对获取后40%血管区域的图像进行MIP操作
    mip_img = basic_mip(vess_img, slice_num=slice_num)
    return mip_img


def creat_dirs(root='', excel=''):
    id_list = pd.read_excel(excel)['Prove-it ID']
    for id in id_list:
        path = os.path.join(root, id)
        if not os.path.exists(path):
            os.mkdir(path)


def get_path_list(root='D:\\Datasets\\MyRegistration', excel='', modal=''):
    # root = '/home/lxy/Datasets/PRoVe-MIP2d'
    id_list = pd.read_excel(excel)['Prove-it ID']
    path_list = []
    for id in id_list:
        path_list.append(os.path.join(root, id, modal))
    return path_list


def mip_all():
    pass


if __name__ == '__main__':
    root = '/homeb/lxy/Datasets/MyRegistration'
    # # id_list = os.listdir(root)
    # problem_id_list = ['1023', '1117', '1119']
    id_list=get_path_list(root=root,excel='/homeb/lxy/MyCode/MTL/info/multi_task_dataset.xlsx')
    # id_list = os.listdir('/homeb/lxy/Datasets/TestSet')

    # legal_list = []
    # for id in id_list:
    #     if id not in problem_id_list:
    #         legal_list.append(id)

    # print(len(legal_list))
    # exit(0)
    src = 'lxy_reg_mCTA1.nii.gz'
    # mask = 'lxy_reg_MCA.nii.gz'
    save = 'lxy_mni_ratio_mip_mCTA1.nii.gz'

    for id in id_list:
        #print(id)
        src_img = nib.load(os.path.join(id, src)).get_fdata()
        src_img = src_img.transpose(0, 2, 1)
        mip = ratio_mip(src_img)
        nib.Nifti1Image(mip, affine=None).to_filename(os.path.join(id, save))
        print(os.path.join(id, save))
        # src_img = nib.load(os.path.join(root, id, src)).get_fdata()
        # mask_img = nib.load(os.path.join(root, id, mask)).get_fdata()
        # src_img = src_img.transpose(0, 2, 1)
        # mask_img = mask_img.transpose(0, 2, 1)
        # mip = mca_ratio_mip(src_img, mask_img)
        # nib.Nifti1Image(mip, affine=None).to_filename(os.path.join(root, id, save))
        # print(os.path.join(root, id, save))
        # break

    # show_nifi('/homeb/lxy/Datasets/MyRegistration/ProVe-IT-01-001/lxy_mni_mca_ratio_mip_mCTA1.nii.gz')
