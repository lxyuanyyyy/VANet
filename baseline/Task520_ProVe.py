#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
import os
import shutil
from collections import OrderedDict
from multiprocessing import Pool

import SimpleITK as sitk
import pandas as pd
import numpy as np
# from batchgenerators.utilities.file_and_folder_operations import *
# from nnunet.configuration import default_num_threads
from sklearn.model_selection import StratifiedShuffleSplit


def load_save_train(args):
    for idx, img_path in enumerate(args['path']):
        save_path = os.path.join(img_dir, args['pat_id'] + '_' + str(idx).zfill(4) + '.nii.gz')
        itk = sitk.ReadImage(img_path, sitk.sitkFloat32)
        sitk.WriteImage(itk, save_path)
    return args['pat_id']


def load_save_test(args):
    pat_id = int(args.split('/')[-1][6:-7])
    if pat_id > 993:
        pat_id = pat_id - 1064 + 101 + 55
    else:
        pat_id = pat_id - 939 + 101
    save_path = os.path.join(img_dir, 'study_' + str(pat_id) + '_0000.nii.gz')
    shutil.copy(args, save_path)
    return 'study_' + str(pat_id)


if __name__ == "__main__":
    dataset_path = "/home/wyh/Codes/dataset/PRoVe"
    data_info_path = "/home/wyh/Codes/dataset/PRoVe/ProveIt_Select_Sheet0505.xlsx"

    output_folder = "/home/wyh/Codes/CoTrClassification/nnUNet/nnU_data/nnUNet_raw_data_base/nnUNet_raw_data/Task520_ProVe"
    # img_dir = join(output_folder, "imagesTr")

    # maybe_mkdir_p(img_dir)

    df = pd.read_excel(data_info_path)[['Prove-it ID', 'Collaterals (1:Good, 2:inter, 3:poor)']]
    df.columns = ['id', 'label']
    data_info = []
    count_id = 1
    train_ids = []
    label = []

    for index, row in df.iterrows():
        tmp = {}
        pat_id = row['id'] + '_' + str(index+1)
        tmp['label'] = row['label'] - 1
        tmp['pat_id'] = pat_id
        tmp['path'] = [os.path.join(dataset_path, row['id'], f"mCTA{i}_brain.nii.gz") for i in range(1, 4)]
        for path in tmp['path']:
            assert os.path.exists(path), path + ' does not exists!'
        train_ids.append(pat_id)
        label.append(row['label'] - 1)
        data_info.append(tmp)


    p = Pool(default_num_threads)
    pat_ids = p.map(load_save_train, data_info)
    p.close()
    p.join()

    json_dict = OrderedDict()
    json_dict['name'] = "COVID-19"
    json_dict['description'] = ""
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "mCTA1",
        "1": "mCTA2",
        "2": "mCTA3"
    }

    json_dict['labels'] = {
        "0": "0",
        "1": "1",
        "2": "2"
    }

    json_dict['numTraining'] = len(train_ids)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s.nii.gz" % ids, "label": '%s' % str(label)}
                             for ids, label in zip(train_ids, label)]
    json_dict['test'] = []

    with open(os.path.join(output_folder, "dataset.json"), 'w') as f:
        json.dump(json_dict, f, indent=4, sort_keys=True)

    # create a dummy split (patients need to be separated)
    splits = [OrderedDict()]
    train_list, val_list = [], []
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=1)
    for train_idx, val_idx in sss.split(train_ids, label):
        train_list, val_list = list(np.array(train_ids)[train_idx]), list(np.array(train_ids)[val_idx])
    splits[0]['train'] = train_list
    splits[0]['val'] = val_list

    save_pickle(splits, join(output_folder, "splits_final.pkl"))
