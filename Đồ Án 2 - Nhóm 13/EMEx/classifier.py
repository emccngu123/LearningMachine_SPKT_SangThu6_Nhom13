import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from sklearn import mixture
# moi cum gom 300 diem du lieu ngau nhien
n_samples = 300

# tao mau du lieu ngau nhien
np.random.seed(0)

# tao du lieu hinh cau nam trong khoang 20 don vi thoi (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])
# tao mot tap du lieu khac cach xa du lieu ban dau
C = np.array([[0., -0.7], [3.5, 3.7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

#  gom 2 tap du lieu lai thanh 1 tap huan luyen duy nhat
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# fit a Gaussian Mixture Model voi components (so luong gaussian la 2)
clf = mixture.GaussianMixture(n_components=2, covariance_type='full')
clf.fit(X_train)

# Hien thi so luong du doan cua mo hinh duoi dang do thi 
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0),
                 levels=np.logspace(0, 3, 4))
CB = plt.colorbar(CS, shrink=0.8, extend='both')
plt.scatter(X_train[:, 0], X_train[:, 1], .8)

plt.title('Negative log-likelihood predicted by a GMM')
plt.axis('tight')
plt.show()