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
print(data)

le = LabelEncoder()
data['pitch_hand'] = le.fit_transform(data['pitch_hand'])

data = data.to_numpy()
y_raw = np.zeros((data.shape[0],1))

X = data[:, [0, 2, 3]]
print("X shape: " + str(X.shape))
print("X: " + str(X))

y_raw = data[:,1]
y_raw = y_raw.reshape(-1,1)

encoder = LabelEncoder()
encoder.fit(y_raw)
pitch_types = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))
print(pitch_types)
y = encoder.transform(y_raw)
y = y.reshape(-1,1)
print("y shape: " + str(y.shape))
print("y: " + str(y))

def predict_ovo(X_test,weights,biases):
    y_pred = np.zeros((X_test.shape[0],))
    num_classes = 21
    pred = np.zeros((X_test.shape[0],num_classes))
    votes = np.zeros((7,))
    for i in range(pred.shape[0]):
        for j in range(pred.shape[1]):
            pred[i][j] = X_test[i][0]*weights[j][0] + X_test[i][1]*weights[j][1] + X_test[i][2]*weights[j][2] + biases[j]
            if pred[i][j] > 1 and j < 6:
                votes[0] += 1
            elif pred[i][j] < 1 and j == 0:
                votes[1] += 1
            elif pred[i][j] < 1 and j == 1:
                votes[2] += 1
            elif pred[i][j] < 1 and j == 2:
                votes[3] += 1
            elif pred[i][j] < 1 and j == 3:
                votes[4] += 1
            elif pred[i][j] < 1 and j == 4:
                votes[5] += 1
            elif pred[i][j] < 1 and j == 5:
                votes[6] += 1
            elif pred[i][j] > 1 and j >= 6 and j < 11:
                votes[1] += 1
            elif pred[i][j] < 1 and j == 6:
                votes[2] += 1
            elif pred[i][j] < 1 and j == 7:
                votes[3] += 1
            elif pred[i][j] < 1 and j == 8:
                votes[4] += 1
            elif pred[i][j] < 1 and j == 9:
                votes[5] += 1
            elif pred[i][j] < 1 and j == 10:
                votes[6] += 1
            elif pred[i][j] > 1 and j >= 11 and j < 15:
                votes[2] += 1
            elif pred[i][j] < 1 and j == 11:
                votes[3] += 1
            elif pred[i][j] < 1 and j == 12:
                votes[4] += 1
            elif pred[i][j] < 1 and j == 13:
                votes[5] += 1
            elif pred[i][j] < 1 and j == 14:
                votes[6] += 1
            elif pred[i][j] > 1 and j >= 15 and j < 18:
                votes[3] += 1
            elif pred[i][j] < 1 and j == 15:
                votes[4] += 1
            elif pred[i][j] < 1 and j == 16:
                votes[5] += 1
            elif pred[i][j] < 1 and j == 17:
                votes[6] += 1
            elif pred[i][j] > 1 and j >= 18 and j < 20:
                votes[4] += 1
            elif pred[i][j] < 1 and j == 18:
                votes[5] += 1
            elif pred[i][j] < 1 and j == 19:
                votes[6] += 1
            elif pred[i][j] > 1 and j == 20:
                votes[5] += 1
            else:
                votes[6] += 1
        y_pred[i] = np.argmax(votes)
        votes = np.zeros((21,))
    return y_pred

max_accuracy = 0
for iter in range(10):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.10)
    svm_classifier = SVC(kernel='linear',decision_function_shape='ovo')
    svm_classifier.fit(X_train,y_train)
    all_weights = svm_classifier.coef_
    all_biases = svm_classifier.intercept_
    y_pred = predict_ovo(X_test, all_weights, all_biases)
    # y_pred = svm_classifier.predict(X_test)
    # print("y_pred: " + str(y_pred))
    accuracy = accuracy_score(y_pred,y_test)
    if accuracy > max_accuracy:
        max_accuracy = accuracy
        max_w = all_weights
        max_b = all_biases
        confusion = confusion_matrix(y_test,y_pred)
    
print("MAX ACCURACY: " + str(max_accuracy*100))
print("MAX W: " + str(max_w))
print("MAX B: " + str(max_b))


confusion_display = ConfusionMatrixDisplay(confusion,display_labels=["CH","CU","FC","FF","FS","SI","SL"])
confusion_display.plot() 
plt.show()

""""
plt.scatter(x=X_train[:, 0], 
                y=X_train[:, 1], 
                s=0)
# Constructing a hyperplane using a formula.
colors = ["yellow","orange","red","green","blue","purple","black"]
weights = []
bias = []
for i in range(7):
    w = svm_classifier.coef_[i]           # w consists of 2 elements
    weights.append(w)
    b = svm_classifier.intercept_[i]      # b consists of 1 element
    bias.append(b)
    x_points = np.linspace(70, 100)    # generating x-points from -1 to 1
    y_points = -(w[0] / w[1]) * x_points - b / w[1]  # getting corresponding y-points
    # print("y_points Shape: " + str(y_points.shape))
    # Plotting a red hyperplane
    plt.plot(x_points, y_points, c=colors[i])
for i in range(X_train.shape[0]):
    if y_train[i] == 0:
        plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[0],label="CH")
    elif y_train[i] == 1:
        plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[1],label="CU")
    elif y_train[i] == 2:
        plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[2],label="FC")
    elif y_train[i] == 3:
        plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[3],label="FF")
    elif y_train[i] == 4:
        plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[4],label="FS")
    elif y_train[i] == 5:
        plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[5],label="SI")
    elif y_train[i] == 6:
        plt.scatter(X_train[i, 0], X_train[i, 1], color=colors[6],label="SL")
plt.xlabel("Velocity(MPH)")
plt.ylabel("Spin Rate(RPM)")
plt.ylim(0,4000)
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(),loc='upper right')
plt.savefig("./output/SVM_Seven_Pitches")

weights = np.array(weights)
bias = np.array(bias)
# print("Weights: " + str(weights))
# print("Biases: " + str(bias))
# print("w shape: " + str(svm_classifier.coef_.shape))


# def predict_ovo(X, weights, bias):
#     num_classes = 7
#     num_examples = X.shape[0]
#     pred = np.zeros((num_examples, num_classes))
#     index = 0
#     for i in range(num_classes):
#         for j in range(i+1, num_classes):
#             w_ij = weights[index]
#             b_ij = bias[index]
#             index += 1
#             pred_ij = np.dot(X, w_ij.T) + b_ij
#             pred[:, i] += np.where(pred_ij > 0, 1, -1) * (i * 2 - 1)
#             pred[:, j] += np.where(pred_ij > 0, 1, -1) * (j * 2 - 1)
#     y_pred = np.argmax(pred, axis=1)
#     return y_pred
"""




# all_weights = svm_classifier.coef_
# print("all_weights: " + str(all_weights))
# all_biases = svm_classifier.intercept_
# print("all_biases: " + str(all_biases))
# y_new_pred = predict_ovo(X_test, all_weights, all_biases)
# # print("New data predictions: " + str(y_new_pred))
# test_accuracy = accuracy_score(y_new_pred,y_test)
# print("TESTING ACCURACY: " + str(test_accuracy))

# sum = 0
# for i in range(7):
#     for j in range(i+1,7):
#         print("i" + str(i) + " vs. " + str(j))