from PIL import Image
import numpy as np
import WaveletTransform.TwoDWaveletTransformer as wt2d
import Tree.TwoDimBesovTree as tbt
import matplotlib.pyplot as plt


def getNonZeroCoefficients(coeffDict):
    sum = 0
    for coeff in coeffDict.values():
        if coeff["cH"] > 0 or coeff["cV"] > 0 or coeff["cD"] > 0:
            sum += 1
    return sum

def calcMSE(img1, img2):
    return np.mean((img1 - img2) ** 2)


img = Image.open("Koala.jpg").convert("L")
a = np.asarray(img)/255

a_err = a + np.random.normal(0, 0.2, a.shape)
hsm, wt = wt2d.get2DWaveletCoefficients(a_err, "haar")

# cH_dict = {}
# cV_dict = {}
# cD_dict = {}
# for item in wt.items():
#     cH_dict[item[0][0]] = cH_dict.setdefault(item[0][0], []) + [item[1]["cH"]]
#     cV_dict[item[0][0]] = cV_dict.setdefault(item[0][0], []) + [item[1]["cV"]]
#     cD_dict[item[0][0]] = cD_dict.setdefault(item[0][0], []) + [item[1]["cD"]]
#
# print(cH_dict.keys())
# dicts = [["cH", cH_dict], ["cV",cV_dict], ["cD",cD_dict]]
# fig, axs = plt.subplots(1, 3, figsize=(15,5), sharey='all')
# for idx, [name,dic] in enumerate(dicts):
#     for key in dic.keys():
#         axs[idx].set_title(f"values of {name} on different levels")
#         axs[idx].set_yscale("log")
#         axs[idx].set_xlabel("level")
#         axs[idx].set_ylabel("value (logscaled)")
#         axs[idx].plot(len(dic[key])*[key], dic[key],'o',alpha=0.2)
# fig.show()



beta_values = [0.1,0.2,0.3,0.4,0.45,0.475,0.49,0.51,0.55,0.60]
mses = []


for beta in beta_values:
    tree = tbt.TwoDimBesovTree(wt, beta, 8)
    new_coeffs = tree.getMinimizingPosteriorCoefficients()
    # print(list(tree.createConnectedTree().values()).count(1))
    nonzeros = getNonZeroCoefficients(new_coeffs)
    # Image.fromarray(a*255).show()
    new_img = wt2d.inverse2DDWT((hsm, new_coeffs), "haar")
    # print(new_coeffs.values())
    mse = calcMSE(a, new_img)
    # print((new_img[0][0] - a[0][0])**2)
    print(f"mse={calcMSE(a, new_img)} for beta={beta}")
    new_img=new_img
    # Image.fromarray((new_img*255).astype(np.uint8)).show()
    mses.append(mse)

plt.plot(beta_values,mses)
plt.show()

    # Image.fromarray(new_img).show()
# besov_tree = tbt.TwoDimBesovTree(wt, .01, 7)
# new_coeffs = besov_tree.getMinimizingPosteriorCoefficients()

# new_img = wt2d.inverse2DDWT((hsm, new_coeffs), "haar")
#
# print(f"mse {((a - new_img)**2.).mean(axis=None)}")
#
# Image.fromarray(new_img).show()