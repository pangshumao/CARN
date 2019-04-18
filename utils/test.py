from utils.preprocessing import imResize
import matplotlib.pyplot as plt
import scipy.io as sio
imDir = 'J:\\ImageData\\spine\\vgg_aligned\\MR'
data = sio.loadmat('/'.join([imDir, 'case1.mat']))
I = data['I']
I_resize, pixelSizes = imResize(I, 512, 256)
plt.figure('resize')

plt.subplot(121)
plt.title('before resize')
plt.imshow(I,plt.cm.gray)

plt.subplot(122)
plt.title('after resize')
plt.imshow(I_resize,plt.cm.gray)

plt.show()