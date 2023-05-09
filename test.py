import numpy as np
from utils.vis import visualize_depth_numpy, visualize_palette_components_numpy
import imageio
import os


idx = 0
savePath = "/home/ubuntu/Rencq/color_decomposition/example/demo"
opaque_file = f"/home/ubuntu/Rencq/color_decomposition/logs/fruit/testset_vis_019999/plt_opaque_{idx:03d}.npy"
opaque = np.load(opaque_file)

palette = np.array([[1.,1.,1.],[1.,1.,1.],[1.,1.,1.],[1.,1.,1.],[1.,1.,1.]])

opaque[opaque<=0.3] = 0.
plt_decomp_tmp = visualize_palette_components_numpy(opaque,palette)
plt_decomp_tmp = (plt_decomp_tmp * 255).astype('uint8')
imageio.imwrite(os.path.join(savePath, f'plt_decomp_tmp{idx:03d}.png'),plt_decomp_tmp)