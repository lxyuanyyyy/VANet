import os
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import ants

#将标准MCA模板配准到个体样本上获取MCA标签
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
    

#获取MNI的标准vascular territory template

def get_mca(mca_path,individual_path,save_path):
    #标准mca图像
    mca_mni_img = ants.image_read(mca_path)
    #患者个体图像
    individual_img = ants.image_read(individual_path)
    #配准结果
    outs = ants.registration(individual_img,mca_mni_img,type_of_transfrom='Affine')
    #配准后的mca标签
    individual_mca = outs['warpedmovout']
    #保存
    ants.image_write(individual_mca,save_path)

if __name__=='__main__':
    individual_path = 'prepocess\individual_sample.nii.gz'
    vess_territory_template = 'prepocess\mni_vascular_territories.nii.gz'
    save_path = 'prepocess\individual_mca.nii.gz'
    get_mca(mca_path=vess_territory_template,
            individual_path=individual_path,
            save_path=save_path)
        