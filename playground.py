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