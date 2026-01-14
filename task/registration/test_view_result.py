import os
import sys

import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.realpath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, "../../"))
sys.path.append(parent_dir)

from deeperhistreg.dhr_input_output.dhr_loaders import OpenSlideLoader, tiff_loader, LoadMode

he2ihc_loader = tiff_loader.TIFFLoader(
    image_path="../../test/he2ihc_output/warped_source.tiff",
    mode=LoadMode.NUMPY,
)
ihc2he_loader = tiff_loader.TIFFLoader(
    image_path="../../test/ihc2he_output/warped_source.tiff",
    mode=LoadMode.NUMPY,
)


he2ihc = he2ihc_loader.load_level(level=0)
ihc2he = ihc2he_loader.load_level(level=0)


plt.figure(dpi=200)
plt.imshow(he2ihc)
plt.axis('off')
plt.savefig('../../test/he2ihc_result.png', bbox_inches='tight', pad_inches=0)
plt.close()

plt.figure(dpi=200)
plt.imshow(ihc2he)
plt.axis('off')
plt.savefig('../../test/target_result.png', bbox_inches='tight', pad_inches=0)
plt.close()
