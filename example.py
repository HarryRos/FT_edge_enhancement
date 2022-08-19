import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

hexImg = (plt.imread('big_hex.png')[:527, :527, :3].sum(axis=2) / 3. > 0.5).astype(float)
hexImg -= hexImg.mean()  

sampleImg = (plt.imread('dots.png')[:527, :527, :3].sum(axis=2) / 3.).astype(float)
sampleImg -= sampleImg.mean()  

HexImg = np.fft.fftshift(np.fft.fft2(hexImg))
HexImgPower = np.abs(HexImg)**2
HexImgLogPower = np.log10(HexImgPower/HexImgPower.max()) # log power

SampleImg = np.fft.fftshift(np.fft.fft2(sampleImg))
SampleImgPower = np.abs(SampleImg)**2
SampleImgLogPower = np.log10(SampleImgPower/SampleImgPower.max()) # log power

# sampleImg = gaussian_filter(sampleImg, sigma=1, mode='mirror', order=0)

ResImg = np.fft.fft2(hexImg * np.fft.fftshift(np.fft.ifft2(sampleImg)))
ResImgPower = np.abs(ResImg)**2
ResImgLogPower = np.log10(ResImgPower/ResImgPower.max()) # log power

fig, axes = plt.subplots(2, 2)
axes[0, 0].imshow(hexImg, cmap='gray')
axes[0, 1].imshow(HexImgLogPower, vmin=-6, cmap='afmhot')
axes[1, 0].imshow(sampleImg, cmap='gray')
axes[1, 1].imshow(SampleImgLogPower, vmin=-6, cmap='afmhot')
# axes[2, 0].imshow(resImg, cmap='gray')
# for ax in axes:
#     ax.get_xaxis().set_ticks([])
#     ax.get_yaxis().set_ticks([])
plt.figure()
plt.imshow(ResImgLogPower, vmin=-6, cmap='afmhot')
plt.show()
