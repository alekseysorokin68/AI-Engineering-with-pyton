import math
import random

random.seed(42)


# ============================================================
#  Gini Impurity
# ============================================================

def gini_impurity(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


# ============================================================
#  Entropy
# ============================================================

def entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    counts = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return -sum(
        (c / n) * math.log2(c / n) for c in counts.values() if c > 0
    )


# ============================================================
#  Information Gain
# ============================================================

def information_gain(parent_labels, left_labels, right_labels, criterion="gini"):
    measure = gini_impurity if criterion == "gini" else entropy
    n = len(parent_labels)
    n_left = len(left_labels)
    n_right = len(right_labels)
    if n_left == 0 or n_right == 0:
        return 0.0
    parent_impurity = measure(parent_labels)
    child_impurity = (
        (n_left / n) * measure(left_labels) +
        (n_right / n) * measure(right_labels)
    )
    return parent_impurity - child_impurity


# ============================================================
#  Decision Tree
# ============================================================

class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, criterion="gini",
                 max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.tree = None
        self.feature_importances_ = None
        self.n_features = 0
        self.n_samples = 0

    def fit(self, X, y):
        self.n_features = len(X[0])
        self.n_samples = len(X)
        self.feature_importances_ = [0.0] * self.n_features
        self.tree = self._build(X, y, depth=0)
        total = sum(self.feature_importances_)
        if total > 0:
            self.feature_importances_ = [
                fi / total for fi in self.feature_importances_
            ]

    def _build(self, X, y, depth):
        n = len(y)
        n_classes = len(set(y))

        # Stopping conditions
        if (self.max_depth is not None and depth >= self.max_depth) or \
           n < self.min_samples_split or n_classes == 1:
            # Leaf node: return most common class
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return {"leaf": True, "prediction": max(counts, key=counts.get),
                    "n_samples": n, "class_counts": counts}

        # Find best split
        best_gain = -1
        best_feature = None
        best_threshold = None

        features_to_try = list(range(self.n_features))
        if self.max_features and self.max_features < self.n_features:
            features_to_try = random.sample(features_to_try, self.max_features)

        for feature in features_to_try:
            values = sorted(set(X[i][feature] for i in range(n)))
            for k in range(len(values) - 1):
                threshold = (values[k] + values[k + 1]) / 2
                left_labels = [y[i] for i in range(n) if X[i][feature] <= threshold]
                right_labels = [y[i] for i in range(n) if X[i][feature] > threshold]

                if len(left_labels) < self.min_samples_leaf or \
                   len(right_labels) < self.min_samples_leaf:
                    continue

                gain = information_gain(y, left_labels, right_labels, self.criterion)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        if best_gain <= 0:
            counts = {}
            for label in y:
                counts[label] = counts.get(label, 0) + 1
            return {"leaf": True, "prediction": max(counts, key=counts.get),
                    "n_samples": n, "class_counts": counts}

        # Split
        left_idx = [i for i in range(n) if X[i][best_feature] <= best_threshold]
        right_idx = [i for i in range(n) if X[i][best_feature] > best_threshold]

        left_X = [X[i] for i in left_idx]
        left_y = [y[i] for i in left_idx]
        right_X = [X[i] for i in right_idx]
        right_y = [y[i] for i in right_idx]

        # Track feature importance
        weight = n / self.n_samples
        self.feature_importances_[best_feature] += weight * best_gain

        return {
            "leaf": False,
            "feature": best_feature,
            "threshold": best_threshold,
            "gain": best_gain,
            "left": self._build(left_X, left_y, depth + 1),
            "right": self._build(right_X, right_y, depth + 1),
            "n_samples": n,
        }

    def _predict_one(self, x, node):
        if node["leaf"]:
            return node["prediction"]
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return [self._predict_one(x, self.tree) for x in X]

    def accuracy(self, X, y):
        preds = self.predict(X)
        correct = sum(1 for p, t in zip(preds, y) if p == t)
        return correct / len(y)


# ============================================================
#  Random Forest
# ============================================================

class RandomForest:
    def __init__(self, n_trees=100, max_depth=None,
                 min_samples_split=2, max_features="sqrt",
                 criterion="gini"):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.trees = []

    def fit(self, X, y):
        n = len(X)
        n_features = len(X[0])
        mf = None
        if self.max_features == "sqrt":
            mf = max(1, int(math.sqrt(n_features)))
        elif self.max_features == "log2":
            mf = max(1, int(math.log2(n_features)))

        for _ in range(self.n_trees):
            indices = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = [X[i] for i in indices]
            y_boot = [y[i] for i in indices]
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=mf,
                criterion=self.criterion,
            )
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)

    def predict(self, X):
        all_preds = [tree.predict(X) for tree in self.trees]
        predictions = []
        for i in range(len(X)):
            votes = {}
            for preds in all_preds:
                v = preds[i]
                votes[v] = votes.get(v, 0) + 1
            predictions.append(max(votes, key=votes.get))
        return predictions

    def accuracy(self, X, y):
        preds = self.predict(X)
        correct = sum(1 for p, t in zip(preds, y) if p == t)
        return correct / len(y)


# ============================================================
#  Генерация данных
# ============================================================

def generate_data(n_per_class=100):
    X = []
    y = []
    for _ in range(n_per_class):
        X.append([random.gauss(1, 1), random.gauss(1, 1)])
        y.append(0)
    for _ in range(n_per_class):
        X.append([random.gauss(4, 1), random.gauss(4, 1)])
        y.append(1)
    for _ in range(n_per_class):
        X.append([random.gauss(2.5, 1), random.gauss(5, 1)])
        y.append(2)
    combined = list(zip(X, y))
    random.shuffle(combined)
    X, y = zip(*combined)
    return list(X), list(y)


# ============================================================
#  Демо 1: Gini Impurity
# ============================================================

print("=" * 55)
print("ДЕМО 1: Gini Impurity")
print("=" * 55)

print(f"\n[1,1,1,1]:     Gini = {gini_impurity([1,1,1,1]):.4f} (чисто)")
print(f"[0,0,0,1]:     Gini = {gini_impurity([0,0,0,1]):.4f}")
print(f"[0,0,1,1]:     Gini = {gini_impurity([0,0,1,1]):.4f} (50/50)")
print(f"[0,1,2,3]:     Gini = {gini_impurity([0,1,2,3]):.4f} (макс. нечистота)")


# ============================================================
#  Демо 2: Entropy
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 2: Entropy")
print("=" * 55)

print(f"\n[1,1,1,1]:     Entropy = {entropy([1,1,1,1]):.4f} (чисто)")
print(f"[0,0,0,1]:     Entropy = {entropy([0,0,0,1]):.4f}")
print(f"[0,0,1,1]:     Entropy = {entropy([0,0,1,1]):.4f} (50/50)")
print(f"[0,1,2,3]:     Entropy = {entropy([0,1,2,3]):.4f} (макс. неопределённость)")


# ============================================================
#  Демо 3: Information Gain
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 3: Information Gain")
print("=" * 55)

parent = [0, 0, 0, 1, 1, 1]
left_good = [0, 0, 0]
right_good = [1, 1, 1]
left_bad = [0, 0, 1]
right_bad = [1, 1, 0]

ig_good = information_gain(parent, left_good, right_good)
ig_bad = information_gain(parent, left_bad, right_bad)

print(f"\nРодитель: {parent}")
print(f"Хорошее разбиение:   {left_good} | {right_good}  → IG = {ig_good:.4f}")
print(f"Плохое разбиение:    {left_bad} | {right_bad}   → IG = {ig_bad:.4f}")
print(f"\n→ Чем больше IG, тем лучше разбиение")


# ============================================================
#  Демо 4: Decision Tree
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 4: Decision Tree")
print("=" * 55)

X, y = generate_data(100)
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"\nДанные: 300 сэмплов, 3 класса")
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

for depth in [2, 5, 10, None]:
    tree = DecisionTree(max_depth=depth, min_samples_split=2)
    tree.fit(X_train, y_train)
    train_acc = tree.accuracy(X_train, y_train)
    test_acc = tree.accuracy(X_test, y_test)
    depth_str = depth if depth else "без ограничения"
    print(f"\n  max_depth={depth_str}:")
    print(f"    Train accuracy: {train_acc:.4f}")
    print(f"    Test accuracy:  {test_acc:.4f}")
    if depth is None or (depth and depth >= 10):
        print(f"    → Переобучение: train >> test")


# ============================================================
#  Демо 5: Random Forest
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 5: Random Forest")
print("=" * 55)

for n_trees in [1, 5, 10, 50, 100]:
    rf = RandomForest(n_trees=n_trees, max_depth=10, min_samples_split=2)
    rf.fit(X_train, y_train)
    train_acc = rf.accuracy(X_train, y_train)
    test_acc = rf.accuracy(X_test, y_test)
    print(f"\n  n_trees={n_trees:>3d}: Train={train_acc:.4f}, Test={test_acc:.4f}")


# ============================================================
#  Демо 6: Feature Importance
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 6: Feature Importance")
print("=" * 55)

rf_final = RandomForest(n_trees=100, max_depth=10)
rf_final.fit(X_train, y_train)

# Средняя importance по всем деревьям
avg_importance = [0.0] * 2
for tree in rf_final.trees:
    if tree.feature_importances_:
        for i in range(2):
            avg_importance[i] += tree.feature_importances_[i]
total = sum(avg_importance)
if total > 0:
    avg_importance = [fi / total for fi in avg_importance]

print(f"\nСредняя importance по 100 деревьям:")
print(f"  Признак 0 (x): {avg_importance[0]:.4f}")
print(f"  Признак 1 (y): {avg_importance[1]:.4f}")
print(f"\n→ Оба признака важны (данные симметричны)")


# ============================================================
#  Демо 7: Сравнение Tree vs Random Forest
# ============================================================

print("\n" + "=" * 55)
print("ДЕМО 7: Сравнение Tree vs Random Forest")
print("=" * 55)

tree_single = DecisionTree(max_depth=10)
tree_single.fit(X_train, y_train)

rf_final = RandomForest(n_trees=100, max_depth=10)
rf_final.fit(X_train, y_train)

print(f"\n{'Модель':<25} {'Train':<10} {'Test':<10}")
print("-" * 45)
print(f"{'Decision Tree':<25} {tree_single.accuracy(X_train, y_train):.4f}    {tree_single.accuracy(X_test, y_test):.4f}")
print(f"{'Random Forest (100)':<25} {rf_final.accuracy(X_train, y_train):.4f}    {rf_final.accuracy(X_test, y_test):.4f}")
print(f"\n→ RandomForest: train≈test (обобщает лучше)")
print(f"→ Decision Tree: train>>test (переобучается)")
