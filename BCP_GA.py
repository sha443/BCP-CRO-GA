import math
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Read Dataset
WBC_DATASET = "WBC.Cleaned.csv"
BCC_DATASET = "BCC.Cleaned.csv"
BCUCI_DATASET = "BCUCI.Cleaned.csv"

df = pd.read_csv(BCC_DATASET)

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
        population = single_point_crossover(population)
        if np.random.rand() < 0.3:
            population = flip_mutation(population)
            fitness = get_fitness(data, feature_list, target, population)
        if max(fitness) > optimal_value:
            optimal_value = max(fitness)
            optimal_solution = population[np.where(
                fitness == optimal_value)][0]
    return optimal_solution, optimal_value


# Execute Genetic Algorithm to obtain Important Feature
iterations = 1000
feature_set, acc_score = GA(df, feature_list, target, 20, iterations)
# Filter Selected Features
feature_set = [feature_list[i]
               for i in range(len(feature_list)) if feature_set[i] == 1]
# Print List of Features
print('Optimal Feature Set\n', feature_set,
      '\nOptimal Accuracy =', round(acc_score * 100), '%')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=7)

# cf = GradientBoostingClassifier(random_state=0)
# cf = DecisionTreeClassifier(criterion="entropy", max_depth=33)
# cf = XGBClassifier(max_depth = 3,  scale_pos_weight=1)
# cf = RandomForestClassifier(n_estimators=100)
cf = svm.SVC(kernel='linear', C=1000)
cf.fit(X_train, y_train)


predictions = cf.predict(X_test)
cf_pred = cf.predict(X_test)
cf_score = accuracy_score(y_test, cf_pred)
precission_cf = precision_score(y_test, cf_pred)
recall_cf = recall_score(y_test, cf_pred)
f1_cf = f1_score(y_test, cf_pred)
print("Precission : ", precission_cf)
print("Recall : ", recall_cf)
print("F1 : ", f1_cf)
