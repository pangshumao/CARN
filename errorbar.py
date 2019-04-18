import numpy
import os
from matplotlib import pyplot as plt

outDir = 'I:\\spine\\Medical_Image_Analysis\\Figures'

x = [0.5, 1, 1.5, 2]
figSize = [18,8]
plt.figure(figsize=figSize)
'''
*************************************************** IDH ***********************************************
'''
plt.subplot(131)
plt.yticks(fontsize=14)
plt.ylabel('MAE', fontsize=14)

mean_IDH = [1.28, 1.35, 1.27, 1.10]
std_IDH = [1.11, 1.02, 1.08, 0.89]
legend = ['LDH', 'IDD', 'LS', 'Normal']
fig = plt.gca()
fig.axes.get_xaxis().set_visible(False)
fig.spines["top"].set_visible(False)
fig.spines["right"].set_visible(False)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="on", right="off", labelleft="on")

plt.ylim(0, 3)
plt.xlim(0.2, 2.2)
for i in range(len(x)):
    plt.errorbar(x[i], mean_IDH[i], yerr=std_IDH[i], fmt='-o', lolims=True, uplims=True)
plt.legend(legend, loc='upper center', ncol=4, fontsize=10)
plt.title('IDH', fontsize=14)
'''
*************************************************** VBH ***********************************************
'''
plt.subplot(132)
plt.yticks(fontsize=14)
plt.ylabel('MAE', fontsize=14)

mean_IDH = [1.27, 1.31, 1.24, 1.10]
std_IDH = [1.12, 0.99, 1.07, 0.91]
legend = ['LDH', 'IDD', 'LS', 'Normal']
fig = plt.gca()
fig.axes.get_xaxis().set_visible(False)
fig.spines["top"].set_visible(False)
fig.spines["right"].set_visible(False)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="on", right="off", labelleft="on")

plt.ylim(0, 3)
plt.xlim(0.2, 2.2)
for i in range(len(x)):
    plt.errorbar(x[i], mean_IDH[i], yerr=std_IDH[i], fmt='-o', lolims=True, uplims=True)
plt.legend(legend, loc='upper center', ncol=4, fontsize=10)
plt.title('VBH', fontsize=14)
'''
************************************************* Total ***********************************************
'''
plt.subplot(133)
plt.yticks(fontsize=14)
plt.ylabel('MAE', fontsize=14)

mean_IDH = [1.28, 1.33, 1.26, 1.10]
std_IDH = [1.12, 1.01, 1.08, 0.90]
legend = ['LDH', 'IDD', 'LS', 'Normal']
fig = plt.gca()
fig.axes.get_xaxis().set_visible(False)
fig.spines["top"].set_visible(False)
fig.spines["right"].set_visible(False)
plt.tick_params(axis="both", which="both", bottom="off", top="off",
                labelbottom="on", left="on", right="off", labelleft="on")

plt.ylim(0, 3)
plt.xlim(0.2, 2.2)
for i in range(len(x)):
    plt.errorbar(x[i], mean_IDH[i], yerr=std_IDH[i], fmt='-o', lolims=True, uplims=True)
plt.legend(legend, loc='upper center', ncol=4, fontsize=10)
plt.title('Total', fontsize=14)

# plt.show()

plt.savefig(os.sep.join([outDir, 'disease_resluts.tiff']), dpi=600)

