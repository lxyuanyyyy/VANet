import os

import numpy as np
import pandas as pd
import ants
import time
import matplotlib.pyplot as plt

reg_types = ['simlarity',  # 7个自由度
             'affine',  #
             'elastic',  # 弹性变换
             ]


# 配准单模态和mca标签
def my_reg(fix_path, move_path, label_path, save_path, save_label_path):
    # 配准模板
    fix_img = ants.image_read(fix_path)
    # 需要处理的文件
    move_img = ants.image_read(move_path)
    move_label_img = ants.image_read(label_path)

    outs = ants.registration(fix_img, move_img, type_of_transform='Affine')
    # 获取配准后的数据，并保存
    reg_img = outs['warpedmovout']
    ants.image_write(reg_img, save_path)

    # 获取move到fix的转换矩阵；将其应用到 move_label上；插值方式选取 最近邻插值; 这个时候也对应的将label变换到 配准后的move图像上
    reg_label_img = ants.apply_transforms(fix_img, move_label_img, transformlist=outs['fwdtransforms'],
                                          interpolator='nearestNeighbor')

    ants.image_write(reg_label_img, save_label_path)
    print(save_label_path.split('/')[-2], ' save finish.')


# 多文件配准，包括多阶段CTA mca标签文件
def multi_modal_reg(fix_path='/homeb/lxy/MyCode/MTL/prepocess/mni152.nii.gz',
                    modalities=None,
                    save_modalities=None,
                    id_list=None,
                    src_root='/homeb/lxy/Datasets/PRoVe',
                    save_root='/homeb/lxy/Datasets/AntspyReg'):
    if modalities is None:
        modalities = ['cta00.nii.gz',
                      'cta0.nii.gz',
                      'cta1.nii.gz',
                      'atlasTerritory_f3d.nii.gz']
    if save_modalities is None:
        save_modalities = ['lxy_reg_mCTA00.nii.gz',
                           'lxy_reg_mCTA0.nii.gz',
                           'lxy_reg_mCTA1.nii.gz',
                           'lxy_reg_MCA.nii.gz']
    # 配准模板
    fix_img = ants.image_read(fix_path)
    for id in id_list:

        # #文件是否存在
        # for mod in modalities:
        #     path = os.path.join(src_root,id,mod)
        #     if os.path.isfile(path) is False:
        #         print(id,mod)

        st = time.time()
        # 若文件夹不存在，则创建一个
        xxx = os.path.join(save_root, id)
        if not os.path.isdir(xxx):
            os.makedirs(xxx)
        # 以CTA1为配准文件获取映射
        move_img = ants.image_read(os.path.join(src_root, id, modalities[0]))
        # 执行配准
        outs = ants.registration(fix_img, move_img, type_of_transform='Affine')
        # 配准完的数据
        reg_img = outs['warpedmovout']
        # 保存
        ants.image_write(reg_img, os.path.join(save_root, id, save_modalities[0]))

        for i in range(1, len(modalities)):
            img = ants.image_read(os.path.join(src_root, id, modalities[i]))
            reg_img = ants.apply_transforms(fix_img, img, transformlist=outs['fwdtransforms'],
                                            interpolator='nearestNeighbor')
            save_path = os.path.join(save_root, id, save_modalities[i])
            ants.image_write(reg_img, save_path)
            print(save_path)
        end = time.time()

        #print(id, 'reg finish: ', end - st)


def img_show(img, title=''):
    plt.imshow(img, cmap='gray')
    plt.title(title)
    plt.show()


def get_path_list(root='/homeb/lxy/Datasets/PRoVe', excel='', modality='mCTA1_brain.nii.gz'):
    # root = '/home/lxy/Datasets/PRoVe-MIP2d'
    id_list = pd.read_excel(excel)['Prove-it ID']
    path_list = []
    for id in id_list:
        path_list.append(os.path.join(root, id, modality))
    return path_list


def get_id_list(excel=''):
    # root = '/home/lxy/Datasets/PRoVe-MIP2d'
    id_list = pd.read_excel(excel)['Prove-it ID']

    return id_list


def get_sub_dir_list(root: str = '/homeb/lxy/Datasets/DataCalgary/INTERRSeCT'):
    return os.listdir(root)


if __name__ == '__main__':
    # problem_id_list = ['1023', '1117','1119']
    # id_list=get_id_list(excel='/homeb/lxy/MyCode/MTL/info/multi_task_dataset.xlsx')
    # id_list = os.listdir('/homeb/lxy/Datasets/DataCalgary/INTERRSeCT')
    #
    # legal_list = []
    # for id in id_list:
    #     if id not in problem_id_list:
    #         legal_list.append(id)

    # print(id_list)
    # multi_modal_reg(id_list=legal_list,
    #                 modalities=[#'cta00.nii.gz',
    #                             #'cta0.nii.gz',
    #                             'mCTA1_brain.nii.gz',
    #                             'atlasTerritory_f3d.nii.gz'],
    #                 save_modalities=[#'lxy_reg_mCTA00.nii.gz',
    #                                  #'lxy_reg_mCTA0.nii.gz',
    #                                  'lxy_reg_mCTA1.nii.gz',
    #                                  'lxy_reg_MCA.nii.gz'],
    #                 src_root='/homeb/lxy/Datasets/DataCalgary/INTERRSeCT',
    #                 save_root='/homeb/lxy/Datasets/TestSet')

    #id_list = get_id_list(excel='/homeb/lxy/MyCode/MTL/info/multi_task_dataset.xlsx')\
    id_list = ['ProVe-IT-01-002']

    multi_modal_reg(id_list=id_list,
                    modalities=['mCTA1_brain.nii.gz',
                                'mCTA1_vess.nii.gz'],
                    save_modalities=['test.nii.gz',
                                     'vess.nii.gz'],
                    save_root='/homeb/lxy/MyCode/MTL/sample')