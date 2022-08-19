# https://astronomy.stackexchange.com/questions/48836/why-does-the-alignment-evaluation-image-from-jwst-look-like-this#:~:text=A%20quick%20check%20by%20pasting

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

fnames = 'rIUME.png', 'modified_1.png', 'modified_2.png', 'big_hex.png'
imgs = [(plt.imread(fname)[:, :527, :3].sum(axis=2) / 3. > 0.5).astype(float)
        for fname in fnames]

for img in imgs:
    img[:60] = 0. # blank out text

# img = gaussian_filter(img, sigma=1, mode='mirror', order=0) doesn't change conclusion

imgs = [img - img.mean() for img in imgs] # reduces zero frequency strength

# s0, s1 = img.shape
# w = np.hanning(s0)[:, None] * np.hanning(s1) # windowing not necessary in this case

fts = [np.fft.fftshift(np.fft.fft2(img)) for img in imgs]

powers = [np.abs(ft)**2 for ft in fts]
log_powers = [np.log10(p/p.max()) for p in powers] # log power

fig, axes = plt.subplots(len(log_powers), 2)
for img, lp, row in zip(imgs, log_powers, axes):
    row[0].imshow(img, cmap='gray')
    row[1].imshow(lp, vmin=-6, cmap='afmhot')
    for ax in row:
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
plt.show()
