import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

data = pd.read_csv("./input/spin-direction-pitches.csv",usecols=["pitch_hand","api_pitch_type","release_speed","spin_rate"])
# print(data)

le = LabelEncoder()
data['pitch_hand'] = le.fit_transform(data['pitch_hand'])

data = data.to_numpy()
y_raw = np.zeros((data.shape[0],1))

X = data[:, [0, 2, 3]]
# print("X shape: " + str(X.shape))
# print("X: " + str(X))

y_raw = data[:,1]
y_raw = y_raw.reshape(-1,1)


y = np.zeros((y_raw.shape[0],))
for i in range(y_raw.shape[0]):
    if y_raw[i] == "CH" or y_raw[i] == "CU" or y_raw[i] == "SL" or y_raw[i] == "FS":
        y[i] = 0
    else:
        y[i] = 1
# print("y shape: " + str(y.shape))
# print("y: " + str(y))
max_accuracy = 0
max_w = [0, 0, 0]
max_b = 0
for iter in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.10)
    # print("X_train shape: " + str(X_train.shape))
    # print("X_test shape: " + str(X_test.shape))
    svm_classifier = SVC(kernel='linear',decision_function_shape='ovo')
    svm_classifier.fit(X_train,y_train)
    # y_pred = svm_classifier.predict(X_test)
    y_pred_calc = np.zeros((178,1))
    # print("y_pred_calc shape: " + str(y_pred_calc.shape))
    w = svm_classifier.coef_[0]           # w consists of 3 elements
    b = svm_classifier.intercept_[0]      # b consists of 1 element
    for i in range(X_test.shape[0]):
        # print(X_test[i,0])
        # print(X_test[i,1])
        y_pred_calc[i] = X_test[i,0]*w[0] + X_test[i,1]*w[1]  + X_test[i,2]*w[2]+ b
        # print("y_pred_calc: " + str(y_pred_calc[i]))
        if y_pred_calc[i] > 1:
            y_pred_calc[i] = 1
        else:
            y_pred_calc[i] = 0
    accuracy = accuracy_score(y_pred_calc,y_test)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_w = w
        max_b = b
        confusion = confusion_matrix(y_test,y_pred_calc)
print("MAX ACCURACY: " + str(max_accuracy*100))
print("MAX W: " + str(max_w))
print("MAX B: " + str(max_b))

confusion_display = ConfusionMatrixDisplay(confusion,display_labels=["Offspeed","Fastball"])
confusion_display.plot()
plt.show()

# test_accuracy = accuracy_score(y_pred_calc,y_test)
# print("TESTING ACCURACY: " + str(test_accuracy))

# # plt.figure(figsize=(10, 8))
# # Plotting our two-features-space
plt.scatter(x=X_train[:, 0], 
                y=X_train[:, 1], 
                s=8)
# Constructing a hyperplane using a formula.
# w = svm_classifier.coef_[0]           # w consists of 2 elements
print("w shape: " + str(w.shape))
print("w: " + str(w))
# b = svm_classifier.intercept_[0]      # b consists of 1 element
print("b: " + str(b))
x_points = np.linspace(70, 100)    # generating x-points from 70 - 100
y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
print("y_points: "  + str(y_points))
# Plotting a red hyperplane
plt.plot(x_points, y_points, c='green')
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1],
            color='red', marker='o', label='Offspeed')
plt.scatter(X_train[y_train != 0, 0], X_train[y_train != 0, 1],
            color='blue', marker='x', label='Fastball')
plt.scatter(svm_classifier.support_vectors_[:, 0],
            svm_classifier.support_vectors_[:, 1], 
            s=50, 
            facecolors='none', 
            edgecolors='k', 
            alpha=.5)
# Step 2 (unit-vector):
w_hat = svm_classifier.coef_[0] / (np.sqrt(np.sum(svm_classifier.coef_[0] ** 2)))
print("w_hat: " + str(w_hat))
# Step 3 (margin):
margin = 1 / np.sqrt(np.sum(svm_classifier.coef_[0] ** 2))
print("margin: " + str(margin))
# Step 4 (calculate points of the margin lines):
decision_boundary_points = np.array(list(zip(x_points, y_points)))
points_of_line_above = decision_boundary_points + w_hat * margin
points_of_line_below = decision_boundary_points - w_hat * margin
# Plot margin lines
# Blue margin line above
plt.plot(points_of_line_above[:, 0], 
         points_of_line_above[:, 1], 
         'o--', 
         linewidth=2)
# Green margin line below
plt.plot(points_of_line_below[:, 0], 
         points_of_line_below[:, 1], 
         'y--',
         linewidth=2)
plt.ylim(0,4000)
plt.xlabel("Velocity(MPH)")
plt.ylabel("Spin Rate(RPM)")
plt.legend()
plt.savefig("./output/SVM")

