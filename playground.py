from PIL import Image
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovTree as tbt
import matplotlib.pyplot as plt


img = Image.open("Koala.jpg").convert("L")
a = np.asarray(img)/255
hsm, wt = wt2d.get2DWaveletCoefficients(a, "haar")

cH_dict = {}
cV_dict = {}
cD_dict = {}
for item in wt.items():
    cH_dict[item[0][0]] = cH_dict.setdefault(item[0][0], []) + [item[1]["cH"]]
    cV_dict[item[0][0]] = cV_dict.setdefault(item[0][0], []) + [item[1]["cV"]]
    cD_dict[item[0][0]] = cD_dict.setdefault(item[0][0], []) + [item[1]["cD"]]

print(cH_dict.keys())
dicts = [["cH", cH_dict], ["cV",cV_dict], ["cD",cD_dict]]
fig, axs = plt.subplots(1, 3, figsize=(15,5), sharey='all')
for idx, [name,dic] in enumerate(dicts):
    for key in dic.keys():
        axs[idx].set_title(f"values of {name} on different levels")
        axs[idx].set_yscale("log")
        axs[idx].set_xlabel("level")
        axs[idx].set_ylabel("value (logscaled)")
        axs[idx].plot(len(dic[key])*[key], dic[key],'o',alpha=0.2)
fig.show()


# for i in range(1,9):
#     tree = tbt.TwoDimBesovTree(wt, 0.499, i)
#     new_coeffs = tree.getMinimizingPosteriorCoefficients()
#     new_img = wt2d.inverse2DDWT((hsm, new_coeffs), "haar")*255
#     print(f"mse {((a - new_img) ** 2.).mean(axis=None)}")
#     Image.fromarray(new_img).show()
# besov_tree = tbt.TwoDimBesovTree(wt, .01, 7)
# new_coeffs = besov_tree.getMinimizingPosteriorCoefficients()
# #
# new_img = wt2d.inverse2DDWT((hsm, new_coeffs), "haar")
# #
# print(f"mse {((a - new_img)**2.).mean(axis=None)}")
#
# Image.fromarray(new_img).show()