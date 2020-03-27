from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import mglearn
import sys
import scipy as sp
import IPython
from sklearn.neighbors import KNeighborsClassifier

# #################################################################################
# ############################## Iris Dataset #####################################
# # from sklearn.datasets import load_iris
# #
# # iris_dataset = load_iris()
# # #
# # # print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))
# # # print(iris_dataset['DESCR'][:193] + "\n...")
# # # print("Target names: {}".format(iris_dataset['target_names']))
# # # print("Feature names: \n{}".format(iris_dataset['feature_names']))
# # # print("Shape of data: {}".format(iris_dataset['data'].shape))
# #
# # from sklearn.model_selection import train_test_split
# #
# # X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)
# #
# # print("X_train shape: {}".format(X_train.shape))
# # print("y_train shape: {}".format(y_train.shape))
# #
# #
# # # create dataframe from data in X_train
# # # label the columns using the strings in iris_dataset.feature_names
# #
# # iris_dataframe = pd.DataFrame(X_train, columns= iris_dataset.feature_names)
# # print(iris_dataframe)
# # print(y_train)
# # #
# # # # create a scatter matrix from the dataframe, color by y_train
# # # grr = pd.plotting.scatter_matrix(iris_dataframe, c = y_train, figsize = (15,15), marker = "o",
# # #                         hist_kwds = {"bins":20}, s = 60, alpha= 0.9, cmap = mglearn.cm3)
# # # plt.show()
# #
# # from sklearn.neighbors import KNeighborsClassifier
# # knn = KNeighborsClassifier(n_neighbors=1)
# # print(knn.fit(X_train,y_train))
# #
# # x_new = np.array([[5, 2.9, 1, 0.2]])
# # print("X_new.shape: {}".format(x_new.shape))
# # prediction = knn.predict(x_new)
# # print("Prediction: {}".format(prediction))
# # print("Predicted target name: {}".format(iris_dataset['target_names'][prediction]))
# #
# # y_pred = knn.predict(X_test)
# #
# # print("test set prediction \n {}".format(y_pred))
# # print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
# #
# # print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))
# #
# #
# #
#
#
# #################################################################################
# ############################## Chapter 2 ########################################
# # generate dataset
# # X, y = mglearn.datasets.make_forge()
# # # plot dataset
# # print(X[:, 1])
# # mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# # plt.legend(["Class 0", "Class 1"], loc=4)
# # plt.xlabel("First feature")
# # plt.ylabel("Second feature")
# # print("X.shape: {}".format(X.shape))
# #
# # from sklearn.datasets import load_breast_cancer
# # cancer = load_breast_cancer()
# # print('cancer.keys(): \n{}'.format(cancer.keys()))
# # print('Shape of cancer dataset \n {}:'.format(cancer.data.shape))
# # print('Sample counts per class: \n {}'.format(
# #     {n: v for n, v in zip(cancer.target_names, np.bincount(cancer.target))}
# # ))
# # #######################################################
# # from sklearn.datasets import load_boston
# # boston = load_boston()
# # print("Data shape: {}".format(boston.data.shape))
# # X, y = mglearn.datasets.load_extended_boston()
# # print("X.shape: {}".format(X.shape))
# # mglearn.plots.plot_knn_classification(n_neighbors=3)
# #
# #
# #
# # from sklearn.model_selection import train_test_split
# # X,y = mglearn.datasets.make_forge()
# #
# # X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=0)
# #
# # from sklearn.neighbors import KNeighborsClassifier
# # clf = KNeighborsClassifier(n_neighbors=3)
# # clf.fit(X_train, y_train)
# # print("Test prediction: {}".format(clf.predict(X_test)))
# #
# # print("Test set accuracy: {:.2f}".format(clf.score(X_test, y_test)))
# #
# # fig, axes = plt.subplots(1, 3, figsize=(10,3))
# #
# # fig, axes = plt.subplots(1, 3, figsize=(10, 3))
# #
# # for n_neighbors, ax in zip([1, 3, 9], axes):
# #     # the fit method returns the object self, so we can instantiate
# #     # and fit in one line
# #     clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X, y)
# #     mglearn.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=.4)
# #     mglearn.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
# #     ax.set_title("{} neighbor(s)".format(n_neighbors))
# #     ax.set_xlabel("feature 0")
# #     ax.set_ylabel("feature 1")
# # axes[0].legend(loc=3)
# # plt.show()
# #
# #
# # #breast cancer
# #
# # from sklearn.datasets import load_breast_cancer
# #
# #
# # cancer = load_breast_cancer()
# # X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify= cancer.target,
# #                                                     random_state=0)
# #
# #
# # training_accuracy = []
# # test_accuracy= []
# # # try n neighboars from 1 to 10
# # neighboars_settings = range(1,11)
# #
# #
# # for n_neighbors in neighboars_settings:
# #     #build the model
# #     clf = KNeighborsClassifier(n_neighbors= n_neighbors)
# #     clf.fit(X_train, y_train)
# #     # record training set accuracy
# #     training_accuracy.append(clf.score(X_train, y_train))
# #     # record generalisation accuracy
# #     test_accuracy.append(clf.score(X_test,y_test))
# #
# # plt.plot(neighboars_settings, training_accuracy, label = "training_accuracy")
# #
# # plt.plot(neighboars_settings, test_accuracy, label = 'test accuracy')
# # plt.ylabel('Accuracy')
# # plt.xlabel('n_neighboars')
# # plt.legend()
# #
# # plt.show()
# #
# #
# # # Linear models
# #
# # from sklearn.linear_model import LinearRegression
# #
# # X, y = mglearn.datasets.make_wave(n_samples = 60)
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
# #
# # lr = LinearRegression().fit(X_train, y_train)
# #
# # print("lr.coef_: {}".format(lr.coef_))
# # print("lr.intercept_: {}".format(lr.intercept_))
# #
# # print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# # print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# #
# # ##################################################################################
# # X, y = mglearn.datasets.load_extended_boston()
# #
# # print(X,y)
# #
# # X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# # lr = LinearRegression().fit(X_train, y_train)
# #
# # print("Training set score: {:.2f}".format(lr.score(X_train, y_train)))
# # print("Test set score: {:.2f}".format(lr.score(X_test, y_test)))
# #
# # # ridge regression
# #
# # from sklearn.linear_model import Ridge
# #
# # ridge = Ridge().fit(X_train, y_train)
# # print("Training set score: {:.2f}".format(ridge.score(X_train, y_train)))
# # print("Test set score: {:.2f}".format(ridge.score(X_test, y_test)))
# #
# #
# # mglearn.plots.plot_ridge_n_samples()
# # plt.show()
# #
#
# #
# from sklearn.linear_model import LogisticRegression
# from sklearn.svm import LinearSVC
# #
# # X, y = mglearn.datasets.make_forge()
# #
# # fig, axes = plt.subplots(1, 2, figsize = (10, 3))
# #
# # for model, ax in zip([LinearSVC(), LogisticRegression()], axes ):
# #     clf = model.fit(X,y)
# #     mglearn.plots.plot_2d_separator(clf, X, fill = False, eps = 0.5, ax = ax, alpha = 0.7)
# #     mglearn.discrete_scatter(X[:,0], X[:,1], y, ax=ax)
# #     ax.set_title("{}".format(clf.__class__.__name__))
# #     ax.set_xlabel("Feature0")
# #     ax.set_ylabel("Feature1")
# # axes[0].legend()
# #
# # plt.show()
# #
# #
# #
# # from sklearn.datasets import load_breast_cancer
# # cancer = load_breast_cancer()
# # X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, stratify= cancer.target, random_state=42)
# # logreg  = LogisticRegression().fit(X_train,y_train)
# # print("Training set score: {:.3f}".format(logreg.score(X_train,y_train)))
# # print("Test set score: {:.3f}".format(logreg.score(X_test, y_test)))
# #
# #
# #
# # logreg100 = LogisticRegression(C= 100).fit(X_train,y_train)
# # print("Training set score: {:.3f}".format(logreg100.score(X_train, y_train)))
# # print("Test set score: {:.3f}".format(logreg100.score(X_test, y_test)))
#
#
# from sklearn.datasets import make_blobs
# #
#
# X,y = make_blobs(random_state = 42)
# mglearn.discrete_scatter(X[:,0], X[:,1], y)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(["Class 0", "Class 1", "Class 2"])
#
# plt.show()
#
# linear_svm = LinearSVC().fit(X , y)
# print("Coefficient shape:", linear_svm.coef_.shape)
# print("Intercept shape:", linear_svm.coef_.shape)
#
#
# mglearn.discrete_scatter(X[:,0], X[:,1], y)
# line = np.linspace(-15,15)
# for coef, intercept, color in zip(linear_svm.coef_, linear_svm.intercept_, ["b", "r", "g"]):
#     plt.plot(line, -(line * coef[0] + intercept) / coef[1], c = color)
# plt.ylim(-10, 15)
# plt.xlim(-10, 8)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0', 'Line class 1',
#             'Line class 2'], loc=(1.01, 0.3))
#
# plt.show()
# #
# # for coef in linear_svm.coef_:
# #     print(coef)
# #
# # from sklearn.linear_model import LinearRegression
# # import pandas as pd
# #
# ram_prices = pd.read_csv("C:/Users/32638/Desktop/Python_Dash/ram_price.csv")
# #
# # plt.semilogy(ram_prices.date, ram_prices.price)
# # plt.xlabel("Year")
# # plt.ylabel("Price in $/Mbyte")
# #
# # plt.show()
# #
# # from sklearn.tree import DecisionTreeRegressor
#
# data_train = ram_prices[ram_prices.date < 2000]
# data_test = ram_prices[ram_prices.date >= 2000]
#
# X_train = data_train.date[:, np.newaxis]
# # # we use a log-transform to get a simpler relationship of data to target
# # y_train = np.log(data_train.price)
# #
# # tree = DecisionTreeRegressor().fit(X_train, y_train)
# # linear_reg = LinearRegression().fit(X_train, y_train)
# #
# # # predict on all data
# # X_all = ram_prices.date[:, np.newaxis]
# #
# # pred_tree = tree.predict(X_all)
# # pred_lr = linear_reg.predict(X_all)
# #
# # # undo log-transform
# # price_tree = np.exp(pred_tree)
# # price_lr = np.exp(pred_lr)
# #
# #
# # plt.semilogy(data_train.date, data_train.price, label="Training data")
# # plt.semilogy(data_test.date, data_test.price, label="Test data")
# # plt.semilogy(ram_prices.date, price_tree, label="Tree prediction")
# # plt.semilogy(ram_prices.date, price_lr, label="Linear prediction")
# # plt.legend()
# #
# # plt.show()
#
#
# from sklearn.svm import LinearSVC
#
# linear_svm = LinearSVC().fit(X, y)
# #
# # mglearn.plots.plot_2d_separator(linear_svm, X)
# # mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
# #
# # plt.xlabel("Feature 0")
# # plt.ylabel("Feature 1")
#
# plt.show()
#
#
# # add the squared first feature
# X_new = np.hstack([X, X[:, 1:] ** 2])
#
# from mpl_toolkits.mplot3d import Axes3D, axes3d
# figure = plt.figure()
# # visualize in 3D
# ax = Axes3D(figure, elev=-152, azim=-26)
# # plot first all the points with y == 0, then all with y == 1
# mask = y == 0
#
# ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
#            cmap=mglearn.cm2, s=60)
# ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
#           cmap=mglearn.cm2, s=60)
# ax.set_xlabel("feature0")
# ax.set_ylabel("feature1")
# ax.set_zlabel("feature1 ** 2")
# plt.show()
#
#
# ##################################################
# #
# # linear_svm_3d = LinearSVC().fit(X_new, y)
# # coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_
# #
# # # show linear decision boundry
# # figure = plt.figure()
# # ax = Axes3D(figure, elev = 152, azim=-26)
# # xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
# # yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)
# # XX, YY = np.meshgrid(xx, yy)
# # ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
# # ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
# # ax.scatter(X_new[mask, 0], X_new[mask, 1], X_new[mask, 2], c='b',
# #            cmap=mglearn.cm2, s=60)
# # ax.scatter(X_new[~mask, 0], X_new[~mask, 1], X_new[~mask, 2], c='r', marker='^',
# #            cmap=mglearn.cm2, s=60)
# # ax.set_xlabel("feature0")
# # ax.set_ylabel("feature1")
# # ax.set_zlabel("feature0 ** 2")
# #
# # plt.show()
#
#
# # compute the minimum value per feature on the training set
# min_on_training = X_train.min(axis=0)
# # compute the range of each feature (max - min) on the training set
# range_on_training = (X_train - min_on_training).max(axis=0)
# print(range_on_training)
#
#
# line = np.linspace(-3,3,100)
# print(np.tanh(line))
# print(np.maximum(line,0))
#
# plt.plot(line,np.tanh(line), label = "tanh")
# plt.plot(line,np.maximum(line,0), label = "relu")
# plt.legend(loc="best")
# plt.xlabel("x")
# plt.ylabel("relu(x), tanh(x)")
# plt.show()


from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=100, noise=0.25, random_state=3)
# X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)
# mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10), random_state=42).fit(X_train, y_train)
# mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=.3)
# mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
# plt.xlabel("Feature 0")
# plt.ylabel("Feature 1")
# plt.show()


#
#
#
# fig, axes = plt.subplots(2, 4, figsize=(20, 8))
# for axx, n_hidden_nodes in zip(axes, [10, 20]):
#     for ax, alpha in zip(axx, [0.0001, 0.01, 0.1, 1]):
#         mlp = MLPClassifier(solver="lbfgs", random_state=0,
#                             hidden_layer_sizes=[n_hidden_nodes, n_hidden_nodes], alpha=alpha)
#         mlp.fit(X_train, y_train)
#         mglearn.plots.plot_2d_separator(mlp, X_train, fill=True, alpha=0.3, ax=ax)
#         mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
#         ax.set_title("n_hidden= [{},{}]\nalpha={:.4f}".format(n_hidden_nodes, n_hidden_nodes, alpha))
#
# plt.show()


from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

print("Cancer data per-feature: \n{}".format(cancer.data.max(axis=0)))
#
# X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=0)
#
# mlp = MLPClassifier(random_state=42)
# mlp.fit(X_train, y_train)
#
# print("Accuracy on training set: {:.2f}".format(mlp.score(X_train, y_train)))
# print("Accuracy on test set: {:.2f}".format(mlp.score(X_test, y_test)))
#
# # compute the mean value per feature on the training set
# mean_on_train = X_train.mean(axis=0)
# # compute the standard deviation of each feature on the training set
# std_on_train = X_train.std(axis=0)
# # subtract the mean, and scale by inverse standard deviation
# # afterward, mean=0 and std=1
# X_train_scaled = (X_train - mean_on_train) / std_on_train
# # use THE SAME transformation (using training mean and std) on the test set
# X_test_scaled = (X_test - mean_on_train) / std_on_train
#
# mlp = MLPClassifier(random_state=0)
# mlp.fit(X_train_scaled,y_train)
#
# print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
# print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
#
#
#
# mlp = MLPClassifier(max_iter=1000,random_state=0)
# mlp.fit(X_train_scaled,y_train)
#
# print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
# print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
#
#
#
# mlp = MLPClassifier(max_iter=1000,alpha=1,random_state=0)
# mlp.fit(X_train_scaled,y_train)
#
# print("Accuracy on training set: {:.3f}".format(mlp.score(X_train_scaled, y_train)))
# print("Accuracy on test set: {:.3f}".format(mlp.score(X_test_scaled, y_test)))
#
# plt.figure(figsize=(20,5))
# plt.imshow(mlp.coefs_[0], interpolation="none", cmap= "viridis")
# plt.yticks(range(30),cancer.feature_names)
# plt.xlabel("Column in weight matrix")
# plt.ylabel("Input feature")
# plt.colorbar()
# plt.show()


from sklearn.ensemble import GradientBoostingClassifier
from sklearn.datasets import make_blobs, make_circles

X, y = make_circles(noise=0.25, factor=0.5, random_state=1)

# we rename the classes "blue" and "red" for illustration purposes
y_named = np.array(["blue", "red"])[y]

# we can call train_test_split with arbitrarily many arrays;
# all will be split in a consistent manner

X_train, X_test, y_train_named, y_test_named, y_train, y_test = train_test_split(X, y_named, y, random_state=0)

gbrt = GradientBoostingClassifier(random_state=0)
gbrt.fit(X_train, y_train_named)

print("X_test.shape: {}".format(X_test.shape))
print("Decision function shape: {}".format(
    gbrt.decision_function(X_test).shape))

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=1)
print(X_train.shape)
print(X_test.shape)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

print(scaler.fit(X_train))

# transform the data
X_train_scaled = scaler.transform(X_train)

# print dataset properties before and after scaling
print("transformed shape: {}".format(X_train_scaled.shape))
print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))
print("per-feature minimum after scaling:\n {}".format(
    X_train_scaled.min(axis=0)))
print("per-feature maximum after scaling:\n {}".format(
    X_train_scaled.max(axis=0)))

# transform test data
X_test_scaled = scaler.transform(X_test)
# print test data proporties after scaling


fig, axes = plt.subplots(15, 2, figsize=(10, 20))
malignant = cancer.data[cancer.target == 0]
benign = cancer.data[cancer.target == 1]
ax = axes.ravel()

for i in range(30):
    _, bins = np.histogram(cancer.data[:, 1], bins=50)
    ax[i].hist(malignant[:, i], bins=bins, color=mglearn.cm3(0), alpha=.5)
    ax[i].hist(benign[:, i], bins=bins, color=mglearn.cm3(2), alpha=.5)
    ax[i].set_title(cancer.feature_names[i])
    ax[i].set_yticks(())
ax[0].set_xlabel("Feature magnitude")
ax[0].set_ylabel("Frequency")
ax[0].legend(["malignent", "benign"], loc="best")
fig.tight_layout
plt.show()

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(cancer.data)
X_scaled = scaler.transform(cancer.data)

from sklearn.decomposition import PCA

# keep the first two principal components of the data
pca = PCA(n_components=2)
# fit PCA model to breast cancer data
pca.fit(X_scaled)

# transform data onto the first two principal components
X_pca = pca.transform(X_scaled)
print("Original shape: {}".format(str(X_scaled.shape)))
print("Reduced shape: {}".format(str(X_pca.shape)))

# plot first vs. second principal component, colored bc class
plt.figure(figsize=(8, 8))
mglearn.discrete_scatter(X_pca[:, 0], X_pca[:, 1], cancer.target)
plt.legend(cancer.target_names, loc = "best")
plt.gca().set_aspect("equal")
plt.xlabel("First principal component")
plt.ylabel("second principal component")

plt.show()