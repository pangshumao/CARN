import numpy as np
import xlrd
import xlwt
import os
import matplotlib.pyplot as plt
import scipy.io as sio

if __name__ == '__main__':
    fileDir = 'H:\\ImageData\\spine\\uploadData'
    foldDir = 'H:\\ImageData\\spine\\allData\\aligned\\DenseNet-Gan-allData'

    workbook = xlrd.open_workbook(os.sep.join([fileDir, 'information.xlsx']))
    sheet = workbook.sheet_by_index(0)

    diagnosis_cols = sheet.col_values(5)
    diagnosis = diagnosis_cols[1:]

    case_cols = sheet.col_values(0)
    caseInd = np.array(case_cols[1:], dtype=np.int32) - 1

    ldh_ind = []
    idd_ind = []
    ls_ind = []
    normal_ind = []

    for ind, label in enumerate(diagnosis):
        if label == '椎间盘突出':
            ldh_ind.append(ind)
        elif label == '腰椎退行性变':
            idd_ind.append(ind)
        elif label == '滑脱':
            ls_ind.append(ind)
        elif label == '正常':
            normal_ind.append(ind)
        else:
            print('error')

    for i in range(1,6):
        foldData = sio.loadmat(os.sep.join([foldDir, 'fold' + str(i) + '-ind.mat']))
        testInd = foldData['valInd'][0]
        ldh_count = 0
        idd_count = 0
        ls_count = 0
        normal_count = 0

        for ind in testInd:
            if ind in ldh_ind:
                ldh_count += 1
            elif ind in idd_ind:
                idd_count += 1
            elif ind in ls_ind:
                ls_count += 1
            elif ind in normal_ind:
                normal_count += 1
            else:
                print('error')

        fold_count = ldh_count + idd_count + ls_count + normal_count
        ldh_rate = ldh_count / fold_count * 100
        idd_rate = idd_count / fold_count * 100
        ls_rate = ls_count / fold_count * 100
        normal_rate = normal_count / fold_count * 100
        print('fold %d, LDH: %d, IDD: %d, LS: %d, normal: %d, total: %d\n' % (i, ldh_count, idd_count, ls_count, normal_count, fold_count))

        print('fold %d, LDH rate: %.2f, IDD rate: %.2f, LS rate: %.2f, normal rate: %.2f\n' % (i, ldh_rate, idd_rate, ls_rate, normal_rate))
        print('--'*40)

