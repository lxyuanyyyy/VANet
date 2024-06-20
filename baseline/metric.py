import numpy as np
import pandas as pd
import os
from medpy.metric import specificity
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix


def all_scores(y_true, y_pred, prob):
    socres = []
    socres.append(accuracy_score(y_true, y_pred))
    socres.append(precision_score(y_true, y_pred))
    socres.append(recall_score(y_true, y_pred))
    socres.append(f1_score(y_true, y_pred))
    socres.append(specificity(y_pred, y_true))
    socres.append(roc_auc_score(y_true, prob))
    return socres


# if __name__ == '__main__':
#     save_path = '/homec/kuanghl/Codes/ResNet/results/machine/left_right/left_and_right123'
#     models_name = ['svm']
#     for name in models_name:
#         result_path = os.path.join(save_path, f'{name}.npz')
#         results = np.load(result_path)
#         y_true = results['y_true']
#         y_pred = results['y_pred']
#         y_prob = results['y_prob']
#
#         print('Method: ', name)
#         print('ACC:', accuracy_score(y_true, y_pred))
#         print('Precision:', precision_score(y_true, y_pred))
#         print('Recall:', recall_score(y_true, y_pred))
#         print('F1:', f1_score(y_true, y_pred))
#         print('Specificity:', specificity(y_pred, y_true))
#         print('AUC:', roc_auc_score(y_true, y_prob))
#         tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
#         confusion_mat = np.array([[tp, fn], [fp, tn]])
#         print('Confusion matrix: ', confusion_mat)

if __name__  =='__main__':
    save_path = '/homec/kuanghl/Codes/ResNet/results/mip_lasso10/informer/l_all_right/32_40_e-5_20_4_20'
    pred = []
    label = []
    prob = []
    for i in range(2):
        result_path = os.path.join(save_path, f'fold{i}', 'result.csv')
        results = pd.read_csv(result_path)
        pre_temp = results['pred']
        label_temp = results['label']
        prob_path = os.path.join(save_path, f'fold{i}', 'probability.npy')
        prob_tmp = np.load(prob_path)
        pred.extend(pre_temp)
        label.extend(label_temp)
        prob.extend(prob_tmp)

    prob = np.array(prob)
    pred = np.array(pred)
    label = np.array(label)

    print('ACC:', accuracy_score(label, pred))
    print('Precision:', precision_score(label, pred))
    print('Recall:', recall_score(label, pred))
    print('F1:', f1_score(label, pred))
    print('specificity:', specificity(pred, label))
    print('AUC:', roc_auc_score(label, prob[:,1]))
    tn, fp, fn, tp = confusion_matrix(label, pred).ravel()
    # print('specificity:', tn / (fp + tn))
    confusion_mat = np.array([[tp, fn], [fp, tn]])
    print('Confusion matrix: ', confusion_mat)
#
    
    # # scores = []
    # # scores_2 = []
    # # confusion_mats = []
    # y_pred = []
    # y_true = []
    # prob = []
    # pred_list = []
    # porb_list = []
    # confusion_mats = np.zeros(shape=(3, 3))
    # for i in range(5):
    #     results_path = os.path.join(save_path, f'fold{i}', 'result.csv')
    #     result = pd.read_csv(results_path)
    #     pred_tmp = result['y_pred'].values
    #     label_tmp = result['y_true'].values
    #     y_pred.extend(pred_tmp)
    #     y_true.extend(label_tmp)
    #     confusion_mat = confusion_matrix(label_tmp, pred_tmp)
    #     confusion_mats = confusion_mat + confusion_mats
    #     prob_path = os.path.join(save_path, f'fold{i}', 'probability.npy')
    #     prob_tmp = np.load(prob_path)
    #     prob.extend(prob_tmp)
    # 
    # prob = np.array(prob)
    # y_pred = np.array(y_pred)
    # y_true = np.array(y_true)
    # pred_1 = np.where(y_pred > 1, 1, 0)
    # label_1 = np.where(y_true > 1, 1, 0)
    # 
    # print('ACC:', accuracy_score(label_1, pred_1))
    # print('Precision:', precision_score(label_1, pred_1))
    # print('Recall:', recall_score(label_1, pred_1))
    # print('F1:', f1_score(label_1, pred_1))
    # print('specificity:', specificity(label_1, pred_1))
    # print('AUC:', roc_auc_score(label_1, prob[:, 2]))
    # print(confusion_mats)
    # 
    # pred_2 = np.where(y_pred < 1, 1, 0)
    # label_2 = np.where(y_true < 1, 1, 0)
    # 
    # print('ACC:', accuracy_score(label_2, pred_2))
    # print('Precision:', precision_score(label_2, pred_2))
    # print('Recall:', recall_score(label_2, pred_2))
    # print('F1:', f1_score(label_2, pred_2))
    # print('specificity:', specificity(label_2, pred_2))
    # print('AUC:', roc_auc_score(label_2, prob[:, 0]))
    # print(confusion_mats)

    # for i in range(5):
    #     results_path = os.path.join(save_path, f'fold{i}', 'result.csv')
    #     result = pd.read_csv(results_path)
    #     y_pred = result['y_pred'].values
    #     y_true = result['y_true'].values
    #     confusion_mats.append(confusion_matrix(y_true, y_pred))
    #     prob_path = os.path.join(save_path, f'fold{i}', 'probability.npz.npy')
    #     prob = np.load(prob_path)
    #     pred_1 = np.where(y_pred > 1, 1, 0)
    #     label_1 = np.where(y_true > 1, 1, 0)
    #
    #     scores.append(all_scores(label_1, pred_1, prob[:, 2]))
    #
    #     pred_2 = np.where(y_pred < 1, 1, 0)
    #     label_2 = np.where(y_true < 1, 1, 0)
    #     scores_2.append(all_scores(label_2, pred_2, prob[:, 0]))
    #
    #
    #     print('ACC:', accuracy_score(label_1, pred_1))
    #     print('Precision:', precision_score(label_1, pred_1))
    #     print('Recall:', recall_score(label_1, pred_1))
    #     print('F1:', f1_score(label_1, pred_1))
    #     tn, fp, fn, tp = confusion_matrix(label_1, pred_1).ravel()
    #     print('specificity:', tn / (tn + fp))
    #     print('AUC:', roc_auc_score(label_1, prob[:, 2]))
    #
    # scores = np.mean(np.array(scores), axis=0)
    # scores_2 = np.mean(np.array(scores_2), axis=0)
    # print(scores)
    # print(scores_2)
    # print(confusion_mats)


