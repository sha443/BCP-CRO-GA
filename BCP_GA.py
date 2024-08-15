import math
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import MinMaxScaler

# Read Dataset
WBC_DATASET = "data/WBC.Cleaned.csv"
BCC_DATASET = "data/BCC.Cleaned.csv"
BCUCI_DATASET = "data/BCUCI.Cleaned.csv"

df = pd.read_csv(WBC_DATASET)

# d = {'M': 0, 'B': 1}
# df['target'] = df['target'].map(d)

df = df.T.drop_duplicates().T

# Get features and target variables
target = 'Outcome'
feature_list = [i for i in df.columns if i not in target]

X = df.loc[:, df.columns != target]
y = df.loc[:, target]
# Print list of features and target variable names
print('Feature List\n', feature_list, '\n\nTarget = ', target)

# normalize data
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
# print(X[:5])


def init_population(n, c):
    return np.array([[math.ceil(e) for e in pop] for pop in (np.random.rand(n, c) - 0.5)]), np.zeros((2, c)) - 1


def single_point_crossover(population):
    r, c, n = population.shape[0], population.shape[1], np.random.randint(
        1, population.shape[1])
    for i in range(0, r, 2):
        population[i], population[i + 1] = np.append(population[i][0:n], population[i + 1][n:c]), np.append(
            population[i + 1][0:n], population[i][n:c])
    return population


def flip_mutation(population):
    return population.max() - population


def random_selection(population):
    r = population.shape[0]
    new_population = population.copy()
    for i in range(r):
        new_population[i] = population[np.random.randint(0, r)]
    return new_population


def get_fitness(data, feature_list, target, population):
    fitness = []
    for i in range(population.shape[0]):
        columns = [feature_list[j]
                   for j in range(population.shape[1]) if population[i, j] == 1]
        fitness.append(predictive_model(data[columns], data[target]))
    return fitness


def predictive_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=7)

    # rfc = GradientBoostingClassifier(random_state=0)
    rfc = RandomForestClassifier(n_estimators=100,  random_state=42)
    rfc.fit(X_train, y_train)
    return accuracy_score(y_test, rfc.predict(X_test))


def GA(data, feature_list, target, n, max_iter):
    c = len(feature_list)
    population, memory = init_population(n, c)
    fitness = get_fitness(data, feature_list, target, population)
    optimal_value = max(fitness)
    optimal_solution = population[np.where(fitness == optimal_value)][0]
    for i in range(max_iter):
        print('Iteration: ', i)
        population = random_selection(population)
        if np.random.rand() < 0.8:
            population = single_point_crossover(population)
        if np.random.rand() < 0.1:
            population = flip_mutation(population)
            fitness = get_fitness(data, feature_list, target, population)
        if max(fitness) > optimal_value:
            optimal_value = max(fitness)
            optimal_solution = population[np.where(
                fitness == optimal_value)][0]
    return optimal_solution, optimal_value


# Execute Genetic Algorithm to obtain Important Feature
iterations = 10
feature_set, acc_score = GA(df, feature_list, target, 20, iterations)
# Filter Selected Features
feature_set = [feature_list[i]
               for i in range(len(feature_list)) if feature_set[i] == 1]
# Print List of Features
print('Optimal Feature Set\n', feature_set,
      '\nOptimal Accuracy =', round(acc_score * 100), '%')


# keep optimal features
print('Feature Set:', feature_set)
# print('DataFrame Columns:', X.columns)
X_optimal = df.loc[:, np.isin(df.columns, feature_set)]
# print(X_optimal)

X_train, X_test, y_train, y_test = train_test_split(
    X_optimal, y, test_size=0.2, random_state=7)
print("X_train", X_train.shape)
# model = GradientBoostingClassifier(random_state=0)
# model = DecisionTreeClassifier(criterion="entropy", max_depth=33)
# model = XGBClassifier(max_depth=3,  scale_pos_weight=1)
# model = RandomForestClassifier(n_estimators=100)
model = svm.SVC(kernel='linear', C=1000)
history = model.fit(X_train, y_train)

predictions = model.predict(X_test)


# Calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}")

# Calculate precision
precision = precision_score(y_test, predictions, average='weighted')
print(f"Precision: {precision}")

# Calculate recall
recall = recall_score(y_test, predictions, average='weighted')
print(f"Recall: {recall}")

# Calculate F1 score
f1 = f1_score(y_test, predictions, average='weighted')
print(f"F1 Score: {f1}")

# Generate confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Generate classification report
class_report = classification_report(y_test, predictions)
print("Classification Report:")
print(class_report)


# Confusion Matrix
conf_matrix = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()

# ROC Curve and AUC
# Binarize the output labels if it's a multi-class classification problem
# Replace with the actual classes
y_test_binarized = label_binarize(y_test, classes=[0, 1])
predictions_binarized = label_binarize(
    predictions, classes=[0, 1])

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(y_test_binarized.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(
        y_test_binarized[:, i], predictions_binarized[:, i])
    roc_auc[i] = roc_auc_score(
        y_test_binarized[:, i], predictions_binarized[:, i])


# Compute Precision-Recall curve and average precision score for each class
precision = dict()
recall = dict()
average_precision = dict()
for i in range(y_test_binarized.shape[1]):
    precision[i], recall[i], _ = precision_recall_curve(
        y_test_binarized[:, i], predictions_binarized[:, i])
    average_precision[i] = average_precision_score(
        y_test_binarized[:, i], predictions_binarized[:, i])

# Plot Precision-Recall curve
plt.figure()
for i in range(y_test_binarized.shape[1]):
    plt.plot(recall[i], precision[i],
             label=f'Class {i} (AP = {average_precision[i]:.2f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="best")
plt.show()

# Plot ROC curve
plt.figure()
for i in range(y_test_binarized.shape[1]):
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
