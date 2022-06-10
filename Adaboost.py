import numpy as np
import pandas as pd
import sys

class Stump:
    def __init__(self):
        self.polarity = 1
        self.feature_index = None
        self.threshold = None
        self.alpha = None

    def predict(self, x):
        sample_size = x.shape[0]
        x_col = x[:, self.feature_index]
        prediction = np.ones(sample_size)
        if self.polarity != 1:
            prediction[x_col > self.threshold] = -1
        else:
            prediction[x_col < self.threshold] = -1
        return prediction


class Adaboost:
    def __init__(self, k):
        self.k = k
        self.classifier = []

    def fit(self, x, y):
        sample_size, feature_size = x.shape
        weight = np.full(sample_size, (1 / sample_size))
        self.classifier = []
        for i in range(self.k):
            clf = Stump()
            minimum_err = float("inf")
            for f in range(feature_size):
                x_col = x[:, f]
                threshold = np.unique(x_col)
                for t in threshold:
                    p = 1
                    prediction = np.ones(sample_size)
                    prediction[x_col < t] = -1
                    miss_classify = weight[y != prediction]
                    err = sum(miss_classify)
                    if err > 0.5:
                        err,p = 1 - err, -1
                    if err < minimum_err:
                        clf.polarity = p
                        clf.threshold = t
                        clf.feature_index = f
                        minimum_err = err
            eplison = 1e-10
            clf.alpha = 0.5 * np.log((1.0 - minimum_err + eplison) / (minimum_err + eplison))
            prediction = clf.predict(x)

            weight *= np.exp(-clf.alpha * y * prediction)
            weight /= np.sum(weight)
            self.classifier.append(clf)

    def predict(self, x):
        clasification_pred = [clf.alpha * clf.predict(x) for clf in self.classifier]
        y_pred = np.sum(clasification_pred, axis=0)
        y_pred = np.sign(y_pred)
        return y_pred


if __name__ == "__main__":
    from sklearn.model_selection import KFold
    dataset = sys.argv[1]

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

### BREAST CANCER DATA ###

    if dataset == "breast_cancer":
        df = pd.read_csv("breast-cancer-wisconsin.data",
                         names=["ID_Number", "Radius", "Texture", "Perimeter", "Area", "smoothness", "compactness",
                                "concavity", "concave_points", "symmetry", "fractal_dimension"])
        df['compactness'] = np.where(df['compactness'] == "?", 0, df.compactness)
        df['compactness'] = np.where(df['compactness'] == 0, round(df['compactness'].astype(str).astype(int).mean()),
                                     df.compactness)
        df['compactness'] = df['compactness'].astype(int)
        df["fractal_dimension"].replace([2, 4], [-1, 1], inplace=True)

        accuracy_list = []
        k = 2
        model = Adaboost(k)
        for i in range(0, 10):
            df = df.sample(frac=1)
            x, y = df.iloc[:, 1:10].to_numpy(), df.iloc[:, 10].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                x_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(x_train, y_train)
                pred_values = model.predict(X_test)
                acc = accuracy(pred_values, y_test)
                accuracy_list.append(accuracy(pred_values, y_test))
            print("Iteration", i, ":", acc)
            standard_deviation = np.std(accuracy_list)
        print("Accuracy:", sum(accuracy_list) / len(accuracy_list))
        print("Adaboost standard deviation for", sys.argv[1], standard_deviation)

### CAR DATA ###
    elif dataset == "car":
        df = pd.read_csv("car.data", names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "classification"])
        df["classification"].replace(["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace=True)
        df["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace=True)
        df["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace=True)
        df["persons"].replace(["more"], [2], inplace=True)
        df["doors"].replace(["5more"], [5], inplace=True)
        df["maint"].replace(["low", "med", "high", "vhigh"], [1, 2, 3, 4], inplace=True)
        df["buying"].replace(["low", "med", "high", "vhigh"], [1, 2, 3, 4], inplace=True)

        df['doors'] = df['doors'].astype(int)
        df['persons'] = df['persons'].astype(int)

        accuracy_list = []
        k = 2
        model = Adaboost(k)
        for i in range(0,10):
            df = df.sample(frac=1)
            x, y = df.iloc[:, 0:6].to_numpy(), df.iloc[:,6].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                x_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(x_train, y_train)
                pred_values = model.predict(X_test)
                acc = accuracy(pred_values, y_test)
                accuracy_list.append(accuracy(pred_values, y_test))
            print("Iteration",i,":", acc)
            standard_deviation = np.std(accuracy_list)
        print("Accuracy:", sum(accuracy_list)/len(accuracy_list))
        print("Adaboost standard deviation for",sys.argv[1], standard_deviation)


### Letter ###
    elif dataset == "letter":
        df = pd.read_csv("letter-recognition.data",names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar","xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])
        df['lettr'] = [ord(item) - 64 for item in df['lettr']]

        accuracy_list = []
        k = 2
        model = Adaboost(k)
        for i in range(0, 10):
            df = df.sample(frac=1)
            x, y = df.iloc[:, 1:17].to_numpy(), df.iloc[:, 0].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                x_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(x_train, y_train)
                pred_values = model.predict(X_test)
                acc = accuracy(pred_values, y_test)
                accuracy_list.append(accuracy(pred_values, y_test))
            print("Iteration", i, ":", acc)
            standard_deviation = np.std(accuracy_list)
        print("Accuracy:", sum(accuracy_list) / len(accuracy_list))
        print("Adaboost standard deviation for", sys.argv[1], standard_deviation)

### Mushroom ###
    elif dataset == "mushroom":
        df = pd.read_csv("mushroom.data",
                         names=["classification", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",
                                "gill-attachment",
                                "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root",
                                "stalk-surface-above-ring", "stalk-surface-below-ring",
                                "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color",
                                "ring-number", "ring-type", "spore-print-color",
                                "population", "habitat"])
        df["classification"].replace(["e", "p"], [0, 1], inplace=True)
        df["cap-shape"].replace(["b", "c", "x", "f", "k", "s"], [0, 1, 2, 3, 4, 5], inplace=True)
        df["cap-surface"].replace(["f", "g", "y", "s"], [0, 1, 2, 3], inplace=True)
        df["cap-color"].replace(["n", "b", "c", "g", "r", "p", "u", "e", "w", "y"], [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                inplace=True)
        df["bruises"].replace(["t", "f"], [0, 1], inplace=True)
        df["odor"].replace(["a", "l", "c", "y", "f", "m", "n", "p", "s"], [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        df["gill-attachment"].replace(["a", "d", "f", "n"], [0, 1, 2, 3], inplace=True)
        df["gill-spacing"].replace(["c", "w", "d"], [0, 1, 2], inplace=True)
        df["gill-size"].replace(["b", "n"], [0, 1], inplace=True)
        df["gill-color"].replace(["k", "n", "b", "h", "g", "r", "o", "p", "u", "e", "w", "y"],
                                 [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], inplace=True)
        df["stalk-shape"].replace(["e", "t"], [0, 1], inplace=True)
        df["stalk-root"].replace(["b", "c", "u", "e", "z", "r", "?"], [1, 2, 3, 4, 5, 6, 0], inplace=True)
        df["stalk-surface-above-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        df["stalk-surface-below-ring"].replace(["f", "y", "k", "s"], [1, 2, 3, 4], inplace=True)
        df["stalk-color-above-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"],
                                             [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        df["stalk-color-below-ring"].replace(["n", "b", "c", "g", "o", "p", "e", "w", "y"],
                                             [1, 2, 3, 4, 5, 6, 7, 8, 9], inplace=True)
        df["veil-type"].replace(["p", "u"], [1, 2], inplace=True)
        df["veil-color"].replace(["n", "o", "w", "y"], [1, 2, 3, 4], inplace=True)
        df["ring-number"].replace(["n", "o", "t"], [1, 2, 3], inplace=True)
        df["ring-type"].replace(["c", "e", "f", "l", "n", "p", "s", "z"], [1, 2, 3, 4, 5, 6, 7, 8], inplace=True)
        df["spore-print-color"].replace(["k", "n", "b", "h", "r", "o", "u", "w", "y"], [1, 2, 3, 4, 5, 6, 7, 8, 9],
                                        inplace=True)
        df["population"].replace(["a", "c", "n", "s", "v", "y"], [1, 2, 3, 4, 5, 6], inplace=True)
        df["habitat"].replace(["g", "l", "m", "p", "u", "w", "d"], [1, 2, 3, 4, 5, 6, 7], inplace=True)

        accuracy_list =[]
        k = 2
        model = Adaboost(k)
        for i in range(1, 11):
            df = df.sample(frac=1)
            x, y = df.iloc[:, 1:23].to_numpy(), df.iloc[:, 0].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                x_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(x_train, y_train)
                pred_values = model.predict(X_test)
                acc = accuracy(pred_values, y_test)
                accuracy_list.append(accuracy(pred_values, y_test))
            print("Iteration", i, ":", acc)
            standard_deviation = np.std(accuracy_list)
        print("Accuracy:", sum(accuracy_list) / len(accuracy_list))
        print("Adaboost standard deviation for", sys.argv[1], standard_deviation)



    ###  Ecoli ###
    elif dataset == "ecoli":
        df = pd.read_csv("ecoli.data", names=["sequence names", "mcg", "gvh", "lip", "chg",
                                                                "aac", "alm1", "alm2", "decision"],
                                    delim_whitespace=True)
        df["decision"].replace(["cp", "im", "imU", "imS", "imL", "om", "omL", "pp"],
                                          [0, 1, 2, 3, 4, 5, 6, 7], inplace=True)

        df['mcg'] = df['mcg'].astype(int)
        df['gvh'] = df['gvh'].astype(int)
        df['lip'] = df['lip'].astype(int)
        df['chg'] = df['chg'].astype(int)
        df['aac'] = df['aac'].astype(int)
        df['alm1'] = df['alm1'].astype(int)
        df['alm2'] = df['alm2'].astype(int)
        df['decision'] = df['decision'].astype(int)

        accuracy_list = []
        k = 2
        model = Adaboost(k)
        for i in range(1, 11):
            df = df.sample(frac=1)
            x, y = df.iloc[:, 1:8].to_numpy(), df.iloc[:,8].to_numpy()
            y[y == 0] = -1
            kf = KFold(n_splits=5)
            for train_index, test_index in kf.split(x):
                x_train, X_test = x[train_index, :], x[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                model.fit(x_train, y_train)
                pred_values = model.predict(X_test)
                acc = accuracy(pred_values, y_test)
                accuracy_list.append(accuracy(pred_values, y_test))
            print("Iteration", i, ":", acc)
            standard_deviation = np.std(accuracy_list)
        print("Accuracy:", sum(accuracy_list) / len(accuracy_list))
        print("Adaboost standard deviation for", sys.argv[1], standard_deviation)
        print("Adaboost standard deviation for", sys.argv[1], standard_deviation)
