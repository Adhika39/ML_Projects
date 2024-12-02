import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.svm import SVC


wine = load_wine()
wine.target[[10,80,140]]
print(list(wine.target_names))

X = wine.data[:,[0,2]]
y = wine.target

# print("X", X)
# print("y", y)

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state = 0)

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)

X_test_std = sc.transform(X_test)


def versiontuple(v):
    return tuple(map(int, (v.split("."))))

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
    np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0],
        y=X[y == cl, 1],
        alpha=0.6,
        c=cmap(idx),
        edgecolor='black',
        marker=markers[idx],
        label=cl)

        # highlight test samples
        if test_idx:
            if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
                X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
                warnings.warn('Please update to NumPy 1.9.0 or newer')
            else:
                X_test, y_test = X[test_idx, :], y[test_idx]

            plt.scatter(X_test[:, 0],
            X_test[:, 1],
            c='yellow',
            alpha=1.0,
            edgecolor='black',
            linewidths=1,
            marker='o',
            s=55, label='test set')



C_values = [10, 100, 1000, 5000]


for C in C_values:
    model = LogisticRegression(C=C, max_iter = 10000)
    model.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))


    plot_decision_regions(X_combined_std, y_combined, classifier=model, test_idx=range(105, 150))

    plt.xlabel('Alcohol')
    plt.ylabel('Acid')
    plt.title("C=" + str(C))
    plt.legend(loc='upper left')
    plt.tight_layout()
    plt.show()

accuracy = []

model.predict_proba(X_test_std[0, :].reshape(1, -1))
weights, params = [], []
for c in np.arange(-4., 4.):
    model = LogisticRegression(C=10.**c, random_state=0)
    model.fit(X_train_std, y_train)
    weights.append(model.coef_[1])
    params.append(10**c)
    y_pred = model.predict(X_test_std)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    accuracy.append(accuracy_score(y_test, y_pred))

weights = np.array(weights)
plt.plot(params, accuracy, linestyle='--',label='LogisticRegression Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
plt.xscale('log')
plt.show()

gamma_values = [0.1, 10]
accuracy1 = []
accuracy2 = []


for gamma in gamma_values:
    for C in C_values:
        model = SVC(kernel ='rbf', gamma = gamma, C=C)
        model.fit(X_train_std, y_train)

        X_combined_std = np.vstack((X_train_std, X_test_std))
        y_combined = np.hstack((y_train, y_test))

        plot_decision_regions(X_combined_std, y_combined,classifier=model, test_idx=range(105, 150))
        plt.xlabel('Alcohol')
        plt.ylabel('Acid')
        plt.title("C=" + str(C)+", gamma=" + str(gamma))
        plt.legend(loc='upper left')
        plt.tight_layout()
        plt.show()
    if gamma == 0.1:
        model.predict(X_test_std[0, :].reshape(1, -1))
        weights, params = [], []
        for c in np.arange(-4., 4.):
            model = SVC(kernel='rbf', random_state=0, gamma=gamma, C=10.**c)
            model.fit(X_train_std, y_train)
            params.append(10**c)
            y_pred = model.predict(X_test_std)
            print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
            accuracy1.append(accuracy_score(y_test, y_pred))

        weights = np.array(weights)
    else:
        model.predict(X_test_std[0, :].reshape(1, -1))
        weights, params = [], []
        for c in np.arange(-4., 4.):
            model = SVC(kernel='rbf', random_state=0, gamma=gamma, C=10.**c)
            model.fit(X_train_std, y_train)
            params.append(10**c)
            y_pred = model.predict(X_test_std)
            print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
            accuracy2.append(accuracy_score(y_test, y_pred))

        weights = np.array(weights)

#plt.plot(params, accuracy, linestyle='--',label='Logistic Regression')
plt.plot(params, accuracy1, linestyle='--', label="Predictor gamma 0.1")
plt.plot(params, accuracy2, linestyle='--', label="Predictor gamma 10")
plt.ylabel('Accuracy')
plt.xlabel('C')
plt.legend(loc='best')
#plt.title('Model Comparison')
plt.title('SVM Comparison')
plt.xscale('log')
plt.show()
