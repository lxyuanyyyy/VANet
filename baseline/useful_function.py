import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import nibabel as nib
from torch import optim, nn
import torch.optim._functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd
from scipy.ndimage import zoom
import pingouin as pg
from sklearn.metrics import *
import SimpleITK as sitk

'''
matrix=[[3,1,1],
        [2,2,1],
        [0,1,4]]

pre_list=[0,1,2,1,2]
real_list=[1,1,2,2,2]

id_list=[i for i in range(len(pre_list))]
id_list.extend([j for j in range(len(real_list))])
judge=['pre' for i in range(len(pre_list))]
judge.extend(['real' for i in range(len(real_list))])
score_list=pre_list
score_list.extend(real_list)
print(id_list)
print(judge)
print(score_list)

dic={"id":id_list,"judge":judge,"score":score_list}

excel=pd.DataFrame(dic)

icc=pg.intraclass_corr(data=excel,targets='id',raters='judge',ratings='score')

print(icc)
'''
def get_data(dataframe):
    dataframe['Judge1'] = 'pred'
    dataframe['Judge2'] = 'label'
    dataframe['ID'] = dataframe['id']
    columns = ['pat', 'Judge', 'Score']
    n = 2 * len(dataframe)
    result = pd.DataFrame({'ID': np.empty(n, dtype=str), 'Judge': np.empty(n, dtype=str),
                           'Score': np.empty(n, dtype=int)})
    result['ID'][0::2] = dataframe['ID']
    result['ID'][1::2] = dataframe['ID']
    result['Judge'][0::2] = dataframe['Judge1']
    result['Judge'][1::2] = dataframe['Judge2']
    result['Score'][0::2] = dataframe['pred']
    result['Score'][1::2] = dataframe['label']
    result['Score'] = result['Score'].astype('int')
    return result


# 计算icc
def icc_caculate(pred_list, real_list):
    id_list=[]
    judge=[]
    score_list=[]
    shuffle_l=[i for i in range(len(pred_list))]
    random.shuffle(shuffle_l)
    for i in range(len(pred_list)):
        id_list.append(shuffle_l[i])
        judge.append('pred')
        score_list.append(pred_list[shuffle_l[i]])
        id_list.append(shuffle_l[i])
        judge.append('real')
        score_list.append(real_list[shuffle_l[i]])

    '''
    id_list = [i for i in range(len(pred_list))]
    id_list.extend([i for i in range(len(real_list))])
    judge = ['pred' for i in range(len(pred_list))]
    judge.extend('real' for i in range(len(pred_list)))
    score_list = pred_list
    score_list.extend(real_list)
    '''

    dic = {"id": id_list, "judge": judge, "score": score_list}
    excel = pd.DataFrame(dic)

    print(excel)
    #res = get_data(excel)
    #print(res)
    icc = pg.intraclass_corr(data=excel, targets='id', raters='judge', ratings='score')
    return icc


# 混淆矩阵算icc
def matrix_to_icc(matrix):
    pred_list = []
    real_list = []
    # 方阵
    leng = len(matrix)
    # cnt=0
    for i in range(leng):
        for j in range(leng):
            value = matrix[i][j]
            pred_list.extend([j for k in range(value)])
            real_list.extend([i for k in range(value)])
    print(len(pred_list))
    print(len(real_list))
    icc = icc_caculate(pred_list, real_list)
    return icc


def av_icc(icc_path_list):
    pass


# 保存loss图
def save_loss_fig(train_counter, train_loss, test_counter, test_loss, pic_path):
    fig = plt.figure()
    if len(train_counter) != len(train_loss) or len(test_counter) != len(test_loss):
        return -1
    plt.plot(train_counter, train_loss, color='blue')
    plt.scatter(test_counter, test_loss, color='red')
    plt.legend(['train loss', 'test loss'], loc='upper right')
    plt.xlabel('number of training example')
    plt.ylabel('cross entropy loss')
    # 保存图表
    plt.savefig(pic_path)
    print("save fig success!")
    plt.close()
    return 1


# 混淆矩阵算各项分数
def matrix_score(matrix):
    # 还原序列
    pred_list = []
    real_list = []
    # 方阵
    leng = len(matrix)
    # cnt=0
    for i in range(leng):
        for j in range(leng):
            value = matrix[i][j]
            pred_list.extend([j for k in range(value)])
            real_list.extend([i for k in range(value)])
    prec = precision_score(real_list, pred_list, labels=[1], average=None, zero_division=1)
    recall = recall_score(real_list, pred_list, labels=[1], average=None, zero_division=1)
    f1 = f1_score(real_list, pred_list, labels=[1], average=None, zero_division=1)
    return prec, recall, f1


def get_auc(real_list, score_list):
    return roc_auc_score(real_list, score_list)


def ban_random(random_seed=1):
    torch.backends.cudnn.enabled = False
    torch.manual_seed(random_seed)
    torch.manual_seed(0)  # 为CPU设置随机种子
    torch.cuda.manual_seed(0)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(0)  # 为所有GPU设置随机种子
    # random.seed(0)
    np.random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)  # 为了禁止hash随机化，使得实验可复现


# 按比例mip操作
def mip_func(filename, brain_len, ratio=0.25, max_area_idx=0, save_path=''):
    # 用brain_len和ratio计算需要取的切片
    nib_data = nib.load(filename)
    arr3d = nib_data.get_fdata()
    affine = nib_data.affine
    # save_arr=np.ndarray(shape=[512,512,1],dtype=np.float64)
    save_arr = np.zeros([512, 512], dtype=np.float64)
    z_len = arr3d.shape[-1]
    mid = z_len // 2
    half_slice = int(0.5 * ratio * brain_len)
    st = max_area_idx - half_slice
    ed = max_area_idx + half_slice
    # 下标越界判断
    # 修改slice的两
    print('slice num:{}'.format(half_slice * 2))
    if (max_area_idx + half_slice) > z_len or (max_area_idx - half_slice) < 0:
        print("*****************************illegal index!")
        # return -1
        ed = z_len
        st = z_len - (2 * half_slice)

    for i in range(512):
        for j in range(512):
            # print(i,j)
            save_arr[i, j] = max(arr3d[i, j, st:ed])
    print(st, ed)
    nib.Nifti1Image(save_arr, affine).to_filename(save_path)
    # print('save finish')


# 最大脑区取30slice mip操作
def mip30(filename, max_area_idx=0, save_path=''):
    nib_data = nib.load(filename)
    arr3d = nib_data.get_fdata()
    affine = nib_data.affine
    save_arr = np.zeros([512, 512], dtype=np.float64)
    z_len = arr3d.shape[-1]
    half_slice = 15
    st = max_area_idx - half_slice
    ed = max_area_idx + half_slice
    if (max_area_idx + half_slice) > z_len or (max_area_idx - half_slice) < 0:
        ed = z_len
        st = z_len - 2 * half_slice
    for i in range(512):
        for j in range(512):
            # print(i,j)
            save_arr[i, j] = max(arr3d[i, j, st:ed])
    nib.Nifti1Image(save_arr, affine).to_filename(save_path)
    print('save finish.')


def hand_mip(filename, st, ed, save_path):
    nib_data = nib.load(filename)
    arr3d = nib_data.get_fdata()
    affine = nib_data.affine
    save_arr = np.zeros([512, 512, 1], dtype=np.float64)
    for i in range(512):
        for j in range(512):
            save_arr[i, j, 0] = max(arr3d[i, j, st:ed])
    nib.Nifti1Image(save_arr, affine).to_filename(save_path)
    print('save finish')


# 特征提取的固定比例 mip
def mask_mip(filename, mask_path, brain_len, ratio=0.25, max_area_idx=0, save_path=''):
    # 加载mask 提取特征
    mask = nib.load(mask_path).get_fdata()
    # mask中2，4，6，12，14，16部分保存
    nib_data = nib.load(filename)
    arr3d = nib_data.get_fdata()
    affine = nib_data.affine
    save_arr = np.zeros([512, 512], dtype=np.float64)
    z_len = arr3d.shape[-1]
    half_slice = int(0.5 * ratio * brain_len)
    st = max_area_idx - half_slice
    ed = max_area_idx + half_slice
    # 下标越界判断
    if (max_area_idx + half_slice) > z_len or (max_area_idx - half_slice) < 0:
        ed = z_len
        st = z_len - 2 * half_slice
    for i in range(512):
        for j in range(512):
            max_intensity = 0.0
            # 更新最大密度值
            for k in range(st, ed):
                if mask[i, j, k] == 2 or mask[i, j, k] == 4 or mask[i, j, k] == 6 or mask[i, j, k] == 12 or mask[
                    i, j, k] == 14 or mask[i, j, k] == 16:
                    if arr3d[i, j, k] > max_intensity:
                        max_intensity = arr3d[i, j, k]
            save_arr[i, j] = max_intensity
    nib.Nifti1Image(save_arr, affine).to_filename(save_path)
    print('save finish.')


# 对二维数据归一化
def norm(arr2d):
    pass


# 进行连通域分析， 获取最大的两个连通域
def max_connected_domain(itk_mask, save_path):
    """
    获取mask中最大连通域
    :param itk_mask: SimpleITK.Image
    :return:
    """
    cc_filter = sitk.ConnectedComponentImageFilter()
    cc_filter.SetFullyConnected(True)
    output_mask = cc_filter.Execute(itk_mask)

    lss_filter = sitk.LabelShapeStatisticsImageFilter()
    lss_filter.Execute(output_mask)

    num_connected_label = cc_filter.GetObjectCount()  # 获取连通域个数

    area_max_label = 0  # 最大的连通域的label
    area_max = 0

    second_max_label = 0
    second_max_area = 0

    # 连通域label从1开始，0表示背景
    for i in range(1, num_connected_label + 1):
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > area_max:
            area_max_label = i
            area_max = area

    temp = list(range(1, num_connected_label + 1))
    temp.remove(area_max_label)
    for i in temp:
        area = lss_filter.GetNumberOfPixels(i)  # 根据label获取连通域面积
        if area > second_max_area:
            second_max_label = i
            second_max_area = area

    np_output_mask = sitk.GetArrayFromImage(output_mask)

    res_mask = np.zeros_like(np_output_mask)
    res_mask[np_output_mask == area_max_label] = 1
    res_mask[np_output_mask == second_max_label] = 1

    res_itk = sitk.GetImageFromArray(res_mask)
    res_itk.SetOrigin(itk_mask.GetOrigin())
    res_itk.SetSpacing(itk_mask.GetSpacing())
    res_itk.SetDirection(itk_mask.GetDirection())

    #sitk.WriteImage(res_itk, save_path)
    #return res_itk
    res_arr=sitk.GetArrayFromImage(res_itk)
    return res_arr
    # res_itk 是仅有0和1的mask
    #print('save finish')
    # return res_itk


# 特征提取 512x512
def mask_filter(mask, data):
    res_arr = np.zeros([512, 512], dtype=np.float64)
    for i in range(512):
        for j in range(512):
            if mask[i, j] != 0:
                res_arr[i, j] = data[i, j]
            else:
                res_arr[i, j] = 0
    return res_arr


def check_data(list1, list2):
    print('len of list 1:', len(list1))
    print('len of list 2:', len(list2))
    id_list1 = [list1[i][0] for i in range(len(list1))]
    id_list2 = [list2[i][0] for i in range(len(list2))]
    set1 = set(id_list1)
    set2 = set(id_list2)
    print('set 1 len:', len(set1))
    print('set 2 len:', len(set2))
    id_list3 = id_list2 + id_list1
    set3 = set(id_list3)
    print('add len:', len(id_list3))
    # set3=set(id_list3)
    print('all set len:', len(set3))


def zscore_norm(image):
    """ Normalise the image intensity by the mean and standard deviation """
    val_l = 0  # 像素下限
    val_h = 60
    roi = np.where((image >= val_l) & (image <= val_h))
    mu, sigma = np.mean(image[roi]), np.std(image[roi])
    image2 = np.copy(image).astype(np.float32)
    image2[image < val_l] = val_l  # val_l
    image2[image > val_h] = val_h

    eps = 1e-6
    image2 = (image2 - mu) / (sigma + eps)
    return image2


def img_threshold(path, min_val=0,max_val=100):
    nib_data = nib.load(path)
    affine = nib_data.affine
    data = nib_data.get_fdata()
    save_arr = np.clip(data, a_min=min_val, a_max=max_val)
    nib.Nifti1Image(save_arr, affine).to_filename('/home/lxy/threshold_{}_{}.nii.gz'.format(min_val,max_val))
    print('save finish.')



# 混淆矩阵测试ICC
'''
matrix=[[28,5,12],
        [14,14,17],
        [8,6,46]]
matrix2=[[16,29],
         [17,88]]
matrix3=[[27,13,5],
         [13,20,12],
         [5,12,43]]
matrix4=[[34,7,4],
         [17,15,13],
         [3,10,47]]
matrix5=[[24,4,17],
         [8,10,27],
         [10,4,46]]
matrix6=[[31,6,8],
         [13,17,15],
         [10,11,39]]
matrix7=[[12,19,14],
         [8,27,10],
         [0,16,44]]
matrix8=[[28,5,12],
         [14,14,17],
         [8,6,46]]
matrix9=[[16,13,16],
         [10,18,17],
         [7,7,46]]
#print(matrix_score(matrix2))
print(matrix_to_icc(matrix7))
'''
'''
id	label	pred_label	pred_score	model_predict_scores
1537	0	0	0.98361117	[0.98361117 0.01638886]
1548	0	0	0.9303047	[0.9303047  0.06969527]
1540	0	0	0.978964	[0.978964   0.02103605]
15525	1	1	0.9876039	[0.01239602 0.9876039 ]
'''

'''
real_list=[0,0,0,1]
pred_list=[0,0,0,1]
pred_score=[0.98361117,0.9303047,0.978964,0.987603]
model_score=[[0.98361117, 0.01638886],
             [0.9303047,  0.06969527],
             [0.978964,   0.02103605],
             [0.01239602, 0.9876039 ]]

pred=[0.15,0.15,0.15,0.55]
pred1=[[0.6],[0.6],[0.6],[0.6]]

score0=[[0.98361117],[0.9303047],[0.978964],[0.01239602]]
score1=[[0.01638886],  [0.06969527],   [0.02103605], [0.9876039] ]
a1=roc_auc_score(real_list,model_score[:][1])
#a2=roc_auc_score(real_list,score1)
#a3=roc_auc_score(real_list,model_score)
#print(a1)
'''

path='/home/lxy/Datasets/PRoVe-MIP2d/ProVe-IT-01-001/top2_domain_mip_mCTA1.nii.gz'

#img_threshold(path=path,min_val=50,max_val=200)
if __name__=='__main__':
    print(matrix_score([[88,3],[6,53]]))
    print(matrix_score([[81, 10], [4, 55]]))

