import numpy as np
import xlrd
import xlwt
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    outDir = 'H:\\ImageData\\spine\\allData\\DenseNet-Gan-allData\\disc_vb_height\\results\\GCNN-SDNE_depth_8_dataset_SPINE+_gamma_0.5_lambda_g_0.0_lr_0.04_knn_20_laeweight_101.0_sdneweight_0.005'
    workbook = xlrd.open_workbook(os.sep.join([outDir, 'results.xlsx']))
    sheet = workbook.sheet_by_index(0)

    diagnosis_cols = sheet.col_values(5)
    diagnosis = diagnosis_cols[1:]

    total_cols = sheet.col_values(6)
    total_mae = np.array(total_cols[1:])

    disc_cols = sheet.col_values(7)
    disc_mae = np.array(disc_cols[1:])

    vb_cols = sheet.col_values(8)
    vb_mae = np.array(vb_cols[1:])

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

    ldh_disc_mae = disc_mae[ldh_ind]
    ldh_vb_mae = vb_mae[ldh_ind]
    ldh_total_mae = total_mae[ldh_ind]

    idd_disc_mae = disc_mae[idd_ind]
    idd_vb_mae = vb_mae[idd_ind]
    idd_total_mae = total_mae[idd_ind]

    ls_disc_mae = disc_mae[ls_ind]
    ls_vb_mae = vb_mae[ls_ind]
    ls_total_mae = total_mae[ls_ind]

    normal_disc_mae = disc_mae[normal_ind]
    normal_vb_mae = vb_mae[normal_ind]
    normal_total_mae = total_mae[normal_ind]


    labels = ['LDH','IDD', 'LS', 'Normal']
    plt.rcParams['figure.figsize'] = (4.0, 4.0)
    plt.boxplot((ldh_disc_mae, idd_disc_mae, ls_disc_mae, normal_disc_mae), labels=labels, showmeans=True, meanline=True, whis=10)
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.title('IDH', fontsize=14)

    # plt.show()
    plt.savefig(os.sep.join([outDir, 'IDH.tiff']), dpi=600)
    plt.savefig(os.sep.join([outDir, 'IDH.eps']), dpi=600)
    plt.close()

    # plt.rcParams['figure.figsize'] = (4.0, 4.0)
    plt.boxplot((ldh_vb_mae, idd_vb_mae, ls_vb_mae, normal_vb_mae), labels=labels, showmeans=True, meanline=True, whis=10)
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.title('VBH', fontsize=14)

    # plt.show()
    plt.savefig(os.sep.join([outDir, 'VBH.tiff']), dpi=600)
    plt.savefig(os.sep.join([outDir, 'VBH.eps']), dpi=600)
    plt.close()

    # plt.rcParams['figure.figsize'] = (4.0, 4.0)
    plt.boxplot((ldh_total_mae, idd_total_mae, ls_total_mae, normal_total_mae), labels=labels, showmeans=True, meanline=True, whis=10)
    plt.subplots_adjust(left=0.25, right=0.9, top=0.9, bottom=0.2)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('MAE', fontsize=14)
    plt.title('Total', fontsize=14)

    # plt.show()
    plt.savefig(os.sep.join([outDir, 'Total.tiff']), dpi=600)
    plt.savefig(os.sep.join([outDir, 'Total.eps']), dpi=600)
    plt.close()