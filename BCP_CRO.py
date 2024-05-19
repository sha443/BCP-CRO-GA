import random
import pandas
import numpy as np
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict


class CRO:

    def __init__(self):
        global NumHit, T_NumHit, MinStruct, MinPE, MinHit, buffer, KELossRate, MoleColl, alpha, beta
        global TotalMolecule, PE, KE, MoleNumber, molecule, copy_molecules, copy_molecules2
        global X, Y, y, features
        # Create Random Molecules
        MoleNumber = random.randint(1, 570)
        # Create A List of Possible Total Number of Molecules
        molecule = [[0 for i in range(33)] for j in range(MoleNumber)]
        copy_molecules = []
        copy_molecules2 = []
        MinStruct = []
        MinPE = 0
        MinHit = 0
        buffer = 0
        KELossRate = .2
        MoleColl = .6
        alpha = 70
        beta = 33
        TotalMolecule = 0
        T_NumHit = 0
        # Assign Initial Activation Values to Molecules
        for row in range(MoleNumber):
            for j in range(33):
                molecule[row][j] = random.randint(0, 1)

        showtoaltmolecules(self)


def showtoaltmolecules(self):
    # Show All molecules activation values
    # for row in range(MoleNumber):
    #   print(molecule[row])
    print("Initial Molecule Number: " + format(MoleNumber))

    delete_repeatmolecules(self)


def delete_repeatmolecules(self):
    # Create a Copy of Activation Values List
    global copy_molecules, copy_molecules2, molecule, TotalMolecule, MoleNumber
    copy_molecules2 = molecule.copy()
    copy_molecules.clear()
    pos = 0
    count = 0
    # Search molecules with repeat activation values
    for row in range(MoleNumber):
        found = 0
        for row2 in range(row + 1, MoleNumber):
            similar = 0
            all_zero = 0
            for j in range(33):
                if molecule[row][j] == copy_molecules2[row2][j]:
                    similar = similar + 1
                if molecule[row][j] == 0:
                    all_zero = all_zero + 1

            if similar == 33 or all_zero == 33:
                found = 1
                count = count + 1
                break
        if found == 1:
            continue
        copy_molecules.append(molecule[row])
        print(molecule[row])
        pos = pos + 1
    # Delete repeat molecules with same activation values
    print('Repeated and No feature Molecules: ' + format(count))

    molecule = [[]]
    molecule = copy_molecules.copy()
    TotalMolecule = len(molecule)
    print('Remaining Molecules: ' + format(TotalMolecule))

    for row in range(TotalMolecule):
        print(format(row)+'. '+format(molecule[row]))
    print("Molecule Number For CRO: " + format(TotalMolecule))

    read_data(self)


def read_data(self):
    global PE, KE, features, X, y, row, df, NumHit, molecule, copy_molecules, copy_molecules2, TotalMolecule
    global rm_X, tm_X, pm_X, am_X, sm_X, cm_X, conm_X, cpm_X, sym_X, fdm_X, rs_X, ts_X, ps_X, as_X, ss_X, cs_X, cos_X, cps_X, sys_X, fds_X, rw_X, tw_X, pw_X, aw_X, sw_X, cw_X, cow_X, cpw_X, sw_X, fdw_X
    # set datatype for features
    PE = [TotalMolecule]
    KE = [TotalMolecule]
    NumHit = [TotalMolecule]
    features = [33]
    copy_molecules.clear()
    copy_molecules = molecule.copy()

    WBC_DATASET = "WBC.Cleaned.csv"
    BCC_DATASET = "BCC.Cleaned.csv"
    BCUCI_DATASET = "BCUCI.Cleaned.csv"

    df = pandas.read_csv("file:"+BCUCI_DATASET, header=0)
    # df.head()
    # d = {'0': 0, '1': 1}
    # df['Outcome'] = df['Outcome'].map(d)

    # Get all columns except the 'Outcome' column
    columns_to_scale = df.columns[df.columns != 'Outcome']

    scaler = StandardScaler()

    # Loop through each column, reshape, fit and transform
    for column in columns_to_scale:
        column_data = np.array(df[column]).reshape(-1, 1)
        df[column] = scaler.fit_transform(column_data)

    # Display the first few rows of the scaled DataFrame
    df.head()

    c = 0
    # Calcualte PE and KE of all initial molecules using SVM
    for row in range(TotalMolecule):

        features.clear()
        column_names = df.columns
        for j in range(len(column_names)):
            if copy_molecules[row][j] == 1:
                features.append(column_names[j])
        # endfor

        similar = 0
        for j in range(33):
            if copy_molecules[row][j] == 0:
                similar = similar + 1
        c = c + 1
        print('{}. Current Molecule: {}    Current Features: {}'.format(
            c, copy_molecules[row], features))

        # read dataset
        n = TotalMolecule
        X = df[features]
        y = df['Outcome']

        cal_accuracy(self, X, y)

    CRO_iterations(self)


def cal_accuracy(self, X, y):
    global MinPE, MinStruct, NumHit, PE, KE
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2)
    # XGBoost Classifier
    # clf = XGBClassifier(max_depth = 3,  scale_pos_weight=1)

    # clf = DecisionTreeClassifier(criterion="entropy", max_depth=33)
    # Random Forest Classifier Code
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    # predicted_label = clf.predict(X_test)
    # score = -1 * clf.score(X_test, y_test)

    # clf = svm.SVC(kernel='linear  ', C=1000)
    # SVC_Model.fit(X_train, y_train)
    # predicted_label = clf.predict(X_test)
    # print(X.columns)
    # X.fillna(X.mean())
    # y.fillna(y.mean())
    # for col in X.columns:
    #    print(X[col])

    scores = cross_val_score(estimator=clf, X=X, y=y,
                             cv=10, scoring='accuracy')
    predicted_label = cross_val_predict(estimator=clf, X=X, y=y, cv=10)
    score = round(scores.mean() * 100, 4)
    # SVM Classifier Code
    # SVC_Model = SVC()
    # SVC_Model.fit(X_train, y_train)
    # predicted_label = SVC_Model.predict(X_test)
    # score = -1 * SVC_Model.score(X_test, y_test)

    Accuracy_Score = accuracy_score(y, predicted_label)
    Precision_Score = precision_score(y, predicted_label, average="macro")
    Recall_Score = recall_score(y, predicted_label, average="macro")
    F1_Score = f1_score(y, predicted_label, average="macro")

    # print('Average Precision: %0.2f  %%' % (Precision_Score * 100))
    # print('Average Recall: %0.2f  %%' % (Recall_Score * 100))
    # print('Average F1-Score: %0.2f  %%' % (F1_Score * 100))

    cm = np.array(confusion_matrix(y, predicted_label))

    confusion = pd.DataFrame(cm, index=['ES', 'NE'], columns=['ES', 'NE'])
    # try:
    # sns.heatmap(confusion, annot=True, fmt='g')
    # except ValueError:
    # raise ValueError("Confusion matrix values must be integers.")

    CM = confusion_matrix(y, predicted_label)
    print(confusion)
    print(classification_report(y, predicted_label))
    print('Average Accuracy: %0.2f %%  Average Precision: %0.2f  %%   Average Recall: %0.2f  %%   Average F1-Score: '
          '%0.2f  %%' % (Accuracy_Score * 100, Precision_Score * 100, Recall_Score * 100, F1_Score * 100))
    print(score)
    # print(Accuracy_Score)
    PE.append(score)
    KE.append(100)
    NumHit.append(0)
    if MinPE < score:
        MinPE = score
        MinStruct.clear()
        MinStruct = features.copy()
    print('MinPE: {} MinStruct: {}'.format(MinPE, MinStruct))
    j = 0


def CRO_iterations(self):
    global molecule, PE, NumHit, buffer, MinStruct, MinPE, MinHit, buffer, TotalMolecule, T_NumHit, T_MinHit, copy_molecules
    T_MinHit = 0
    terminate = 0
    P_MinPE = 0
    MinPE = 0
    count_decom = 0
    count_inter = 0
    count_onwall = 0
    count_synt = 0

    while terminate < 301:
        # Condition For CRO Termination
        t = random.uniform(0, 1)
        # Decides whether to perform unimoleculat or inter molecular reaction
        if t > MoleColl:
            alpha = random.randint(0, 40)
            # Randomly Select a Molecule for Uni Molecular Reaction
            child = random.randint(0, TotalMolecule - 1)

            if (T_NumHit - T_MinHit) > alpha:
                # New Child Molecule 1
                print('                         Decomposition')
                count_decom = count_decom + 1
                copy_molecules = molecule.copy()
                copy_molecules[child][4] = 1
                copy_molecules[child][5] = 1
                copy_molecules[child][6] = 1
                copy_molecules[child][7] = 1
                # New Child Molecule 2
                copy_molecules2 = molecule.copy()
                copy_molecules2[child][0] = 1
                copy_molecules2[child][1] = 1
                copy_molecules2[child][2] = 1
                copy_molecules2[child][3] = 1
                # Calculate PE for Child Molecules
                print('Child1: {} Child2: {}'.format(
                    copy_molecules[child], copy_molecules2[child]))
                acc_result_child_1 = cal_child_accuracy(
                    self, copy_molecules, child)
                acc_result_child_2 = cal_child_accuracy(
                    self, copy_molecules2, child)
                # PE for parent molecule
                acc_result_parent = PE[child]
                sigma_1 = random.uniform(0, 1)
                sigma_2 = random.uniform(0, 1)

                if (acc_result_parent + KE[child] + sigma_1 * sigma_2 * buffer) > (acc_result_child_1 + acc_result_child_2):
                    Edec = PE[child] + KE[child] + sigma_1 * sigma_2 * \
                        buffer - (acc_result_child_1 + acc_result_child_2)
                    sigma_3 = random.uniform(0, 1)
                    KE1 = Edec * sigma_3
                    KE2 = Edec * (1 - sigma_3)
                    buffer = buffer * (1 - sigma_1 * sigma_2)

                    KE[child] = KE1
                    PE[child] = acc_result_child_1
                    molecule[child] = copy_molecules[child].copy()
                    NumHit[child] = 0
                    KE.append(KE2)
                    PE.append(acc_result_child_2)
                    molecule.append(copy_molecules2[child])
                    NumHit.append(0)
                    TotalMolecule = TotalMolecule + 1
                    # New Optimal Features Found and Assign
                    if acc_result_child_1 > MinPE:
                        MinPE = acc_result_child_1
                        MinStruct = copy_molecules[child].copy()
                    elif acc_result_child_2 > MinPE:
                        MinPE = acc_result_child_2
                        MinStruct = copy_molecules2[child].copy()
                    NumHit[child] = 0
                    T_MinHit = T_NumHit
                T_NumHit = T_NumHit + 1

            # on-wall ineffective Collision
            else:
                print('                         On Wall Ineffective Collision')
                count_onwall = count_onwall + 1
                T_NumHit = T_NumHit + 1
                # Random number for molecule selection
                copy_molecules.clear()
                copy_molecules = molecule.copy()
                rand_position = random.randint(0, 7)

                if copy_molecules[child][rand_position] == 0:
                    copy_molecules[child][rand_position] = 1
                else:
                    copy_molecules[child][rand_position] = 0
                # Repair Function Reject zero or one feature
                similar = 0
                for j in range(33):
                    # for j in range(7):
                    if copy_molecules[child][j] == 0:
                        similar = similar + 1
                if similar >= 7:
                    continue
                print('Child: {} '.format(copy_molecules[child]))
                acc_result_child = cal_child_accuracy(
                    self, copy_molecules, child)
                acc_result_parent = PE[child]
                if acc_result_child < acc_result_parent + KE[child]:
                    alpha1 = random.uniform(KELossRate, 1)
                    new_KE = (acc_result_parent -
                              acc_result_child + KE[child]) * alpha1
                    buffer = buffer + \
                        (acc_result_parent - acc_result_child +
                         KE[child]) * (1 - alpha1)
                    PE[child] = acc_result_child
                    KE[child] = new_KE
                    NumHit[child] = NumHit[child] + 1
                    molecule[child][rand_position] = copy_molecules[child][rand_position]

                    if MinPE < acc_result_child:
                        MinPE = acc_result_child
                        MinStruct = copy_molecules[child].copy()
                        MinHit = NumHit[child]
                        T_MinHit = T_NumHit

                # print('On Wall Ineffective End')
                # inter-molecular reaction
        else:
            random_child1 = -1
            random_child2 = -1
            # Select Two Child Molecules Number
            while random_child1 == random_child2:
                random_child1 = random.randint(0, TotalMolecule - 1)
                random_child2 = random.randint(0, TotalMolecule - 1)
            copy_molecules = molecule.copy()
            copy_molecules2 = molecule.copy()
            # Synthesis reaction condition
            if KE[random_child1] <= beta:
                print('                         Synthesis')
                count_synt = count_synt + 1
                copy_molecules[random_child1][4] = molecule[random_child2][4]
                copy_molecules[random_child1][5] = molecule[random_child2][5]
                copy_molecules[random_child1][6] = molecule[random_child2][6]
                copy_molecules[random_child1][7] = molecule[random_child2][7]

                similar = 0
                for j in range(33):
                    if copy_molecules[random_child1][j] == 0:
                        similar = similar + 1
                if similar >= 7:
                    print('Ignore synthesis for 1 or 0 feature: ')
                    print(copy_molecules[random_child1])
                    continue
                similar = 0
                for j in range(33):
                    if copy_molecules[random_child2][j] == 0:
                        similar = similar + 1

                print('Child: {} '.format(copy_molecules[random_child1]))

                acc_result_child_1 = cal_child_accuracy(
                    self, copy_molecules, random_child1)

                acc_result_parent1 = PE[random_child1]
                acc_result_parent2 = PE[random_child2]
                KE_result_parent1 = KE[random_child1]
                KE_result_parent2 = KE[random_child2]
                if (acc_result_parent1 + acc_result_parent2 + KE_result_parent1 + KE_result_parent2) > acc_result_child_1:
                    KE[random_child1] = (acc_result_parent1 + acc_result_parent2 +
                                         KE_result_parent1 + KE_result_parent2) - acc_result_child_1
                    PE[random_child1] = acc_result_child_1
                    NumHit[random_child1] = 0
                    if MinPE < acc_result_child_1:
                        MinPE = acc_result_child_1
                        MinStruct = copy_molecules[random_child1].copy()
                    index = molecule.index(copy_molecules[random_child2])
                    del molecule[index]
                    del NumHit[index]
                    del PE[index]
                    del KE[index]
                    T_MinHit = T_NumHit
                    TotalMolecule = TotalMolecule - 1

                else:
                    NumHit[random_child1] = NumHit[random_child1] + 1
                    NumHit[random_child2] = NumHit[random_child2] + 1
                T_NumHit = T_NumHit + 1

            else:
                print('                         Inter Molecule')
                count_inter = count_inter + 1
                random_position = random.randint(0, 7)
                if copy_molecules[random_child1][random_position] == 1:
                    copy_molecules[random_child1][random_position] = 0
                else:
                    copy_molecules[random_child1][random_position] = 1

                similar = 0
                for j in range(33):
                    if copy_molecules[random_child1][j] == 0:
                        similar = similar + 1
                if similar >= 7:
                    print('Ignore synthesis for 1 or 0 features: ')
                    print(copy_molecules[random_child1])
                    continue
                similar = 0
                for j in range(33):
                    if copy_molecules[random_child2][j] == 0:
                        similar = similar + 1
                if similar >= 7:
                    print('Ignore synthesis for 1 or 0 features: ')
                    print(copy_molecules[random_child2])
                    continue
                print('Child1: {} Child2: {}'.format(
                    copy_molecules[random_child1], copy_molecules[random_child2]))
                acc_result_child_1 = cal_child_accuracy(
                    self, copy_molecules, random_child1)
                acc_result_child_2 = cal_child_accuracy(
                    self, copy_molecules2, random_child2)
                NumHit[random_child1] = NumHit[random_child1] + 1
                NumHit[random_child2] = NumHit[random_child2] + 1
                T_NumHit = T_NumHit + 1
                acc_result_parent1 = PE[random_child1]
                acc_result_parent2 = PE[random_child2]
                KE_result_parent1 = KE[random_child1]
                KE_result_parent2 = KE[random_child2]
                E_inter = (acc_result_parent1 + acc_result_parent2 + KE_result_parent1 + KE_result_parent2) - (
                    acc_result_child_1 + acc_result_child_2)
                if E_inter > 0:
                    Sigma_4 = random.uniform(0, 1)
                    KE1 = E_inter * Sigma_4
                    KE2 = E_inter * (1 - Sigma_4)
                    molecule[random_child1] = copy_molecules[random_child1].copy()
                    molecule[random_child2] = copy_molecules2[random_child2].copy()
                    PE[random_child1] = acc_result_child_1
                    PE[random_child2] = acc_result_child_2
                    KE[random_child1] = KE1
                    KE[random_child2] = KE2
                    NumHit[random_child1] = NumHit[random_child1] + 1
                    NumHit[random_child2] = NumHit[random_child2] + 1

                    if acc_result_child_1 > MinPE and acc_result_child_1 > acc_result_child_2:
                        MinPE = acc_result_child_1
                        MinStruct = molecule[random_child1].copy()
                        MinHit = NumHit[random_child1]
                        T_MinHit = T_NumHit
                    if acc_result_child_2 > MinPE and acc_result_child_1 < acc_result_child_2:
                        MinPE = acc_result_child_2
                        MinStruct = molecule[random_child2].copy()
                        MinHit = NumHit[random_child2]
                        T_MinHit = T_NumHit

                # print('Inter Molecular Ineffective End')
        print('Optimal Accuracy:  {} Overall Accuracy: {}  Optimal Features: {} Terminate: {}'.format(MinPE, P_MinPE,
                                                                                                      MinStruct,
                                                                                                      terminate))
        print(MinStruct)
        if MinPE > P_MinPE:
            terminate = 0
            P_MinPE = MinPE
        else:
            terminate = terminate + 1
    print('Decomposition: {} Synthesis: {} On-Wall: {} Inter-Molecular: {}'.format(
        count_decom, count_synt, count_onwall, count_inter))


def cal_child_accuracy(self, child_Mol, child):
    global features, X, y
    features.clear()
    column_names = df.columns
    for j in range(len(column_names)):
        if copy_molecules[row][j] == 1:
            features.append(column_names[j])
    # endfor

    X = df[features]
    y = df['Outcome']

    # clf = DecisionTreeClassifier(criterion="entropy", max_depth=33)
    # XGBoost Classifier
    clf = XGBClassifier(max_depth=3, scale_pos_weight=1)
    # Random Forest Classifier Code
    # clf = RandomForestClassifier(n_estimators=100)
    # clf.fit(X_train, y_train)

    # predicted_label = clf.predict(X_test)
    # score = -1 * clf.score(X_test, y_test)

    # clf = svm.SVC(kernel='linear  ', C=1000)
    # SVC_Model.fit(X_train, y_train)
    # predicted_label = clf.predict(X_test)
    scores = cross_val_score(estimator=clf, X=X, y=y,
                             cv=10, scoring='accuracy')
    predicted_label = cross_val_predict(estimator=clf, X=X, y=y, cv=10)
    score = round(scores.mean() * 100, 4)
    # score = -1 * SVC_Model.score(X_test, y_test)

    # Code for Confusion Matrix, Recall and Precision

    Accuracy_Score = accuracy_score(y, predicted_label)
    Precision_Score = precision_score(y, predicted_label, average="macro")
    Recall_Score = recall_score(y, predicted_label, average="macro")
    F1_Score = f1_score(y, predicted_label, average="macro")

    # print('Average Precision: %0.2f  %%' % (Precision_Score * 100))
    # print('Average Recall: %0.2f  %%' % (Recall_Score * 100))
    # print('Average F1-Score: %0.2f  %%' % (F1_Score * 100))

    cm = np.array(confusion_matrix(y, predicted_label))

    confusion = pd.DataFrame(cm, index=['ES', 'NE'], columns=['ES', 'NE'])
    # try:
    # sns.heatmap(confusion, annot=True, fmt='g')
    # except ValueError:
    # raise ValueError("Confusion matrix values must be integers.")

    CM = confusion_matrix(y, predicted_label)
    print(confusion)
    print(classification_report(y, predicted_label))
    print('Average Accuracy: %0.2f %%  Average Precision: %0.2f  %%   Average Recall: %0.2f  %%   Average F1-Score: '
          '%0.2f  %%' % (Accuracy_Score * 100, Precision_Score * 100, Recall_Score * 100, F1_Score * 100))
    print(score)
    # print(features)
    return score


if __name__ == "__main__":
    CRO()
