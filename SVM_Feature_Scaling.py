import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt

data = pd.read_csv("./input/spin-direction-pitches.csv",usecols=["pitch_hand","api_pitch_type","release_speed","spin_rate"])
print(data)

pitch_types = data["api_pitch_type"].unique()
mean_velo_dict = {pt: np.mean(data[data["api_pitch_type"] == pt]["release_speed"]) for pt in pitch_types}
mean_spin_dict = {pt: np.mean(data[data["api_pitch_type"] == pt]["spin_rate"]) for pt in pitch_types}
print("Mean velocity dictionary:")
print(mean_velo_dict)

print("Mean spin dictionary:")
print(mean_spin_dict)

le = LabelEncoder()
data['pitch_hand'] = le.fit_transform(data['pitch_hand'])

data = data.to_numpy()
y_raw = np.zeros((data.shape[0],1))

X = data[:, [0, 2, 3]]
print("X shape: " + str(X.shape))
print("X: " + str(X))

mean_velo = np.mean(X[:,1])
mean_spin = np.mean(X[:,2])
print("AVERAGE SPEED: " + str(mean_velo))
print("AVERAGE VELOCITY: " + str(mean_spin))


y_raw = data[:,1]
y_raw = y_raw.reshape(-1,1)

y = np.zeros((y_raw.shape[0],))
for i in range(y_raw.shape[0]):
    if y_raw[i] == "CH" or y_raw[i] == "CU" or y_raw[i] == "SL" or y_raw[i] == "FS":
        y[i] = 0
    else:
        y[i] = 1
print("y shape: " + str(y.shape))

def train_experience_level(X,y,experience_levels):
    for key, values in experience_levels.items():
        for i in range(y.shape[0]):
            for pitch_type, scaled_features in values.items():
                if pitch_type == y_raw[i]:
                    X[i,1] = X[i,1]*scaled_features[0]
                    X[i,2] = X[i,2]*scaled_features[1]
        # print("X scaled " + key + " : " + str(X))
        X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.10)
        svm_classifier = SVC(kernel='linear',decision_function_shape='ovo')
        svm_classifier.fit(X_train,y_train)
        # y_pred = svm_classifier.predict(X_test)
        w = svm_classifier.coef_[0]           # w consists of 2 elements
        b = svm_classifier.intercept_[0]      # b consists of 1 element
        y_pred_calc = np.zeros((178,1))
        for i in range(X_test.shape[0]):
            y_pred_calc[i] = X_test[i,0]*w[0] + X_test[i,1]*w[1]  + X_test[i,2]*w[2]+ b
            if y_pred_calc[i] > 1:
                y_pred_calc[i] = 1
            else:
                y_pred_calc[i] = 0
        accuracy = accuracy_score(y_pred_calc,y_test)
        print("Accuracy " + key + " : " + str(accuracy))
        print("w " + key + " : " + str(w))
        print("b " + key + " : " + str(b))
        
def train_multiple_iterations(X,y,experience_levels,iterations):
    accuracies = {}
    weights = {}
    bias = {}
    best_accuracy = {}
    best_weights = {}
    best_bias = {}
    for age in experience_levels.keys():
        accuracies[age] = []
        weights[age] = []
        bias[age] = []
        best_accuracy[age] = 0
        best_weights[age] = None
        best_bias[age] = None
    # print("accuracies: " + str(accuracies))
    for iter in range(iterations):
        print("Iteration: " + str(iter))
        for key, values in experience_levels.items():
            if key != "10":
                continue
            else:
                X_scaled = np.zeros_like(X)
                X_scaled[:,0] = X[:,0]
                for i in range(X.shape[0]):
                    for pitch_type, scaled_features in values.items():
                        if pitch_type == y_raw[i]:
                            X_scaled[i,1] = X[i,1]*scaled_features[0]
                            X_scaled[i,2] = X[i,2]*scaled_features[1]
                if key == "10" or key == "15":
                    print("X scaled " + key + " : " + str(X_scaled))
                X_train, X_test, y_train, y_test = train_test_split(X_scaled,y,test_size=.10)
                svm_classifier = SVC(kernel='linear',decision_function_shape='ovo')
                svm_classifier.fit(X_train,y_train)
                # y_pred = svm_classifier.predict(X_test)
                w = svm_classifier.coef_[0]
                b = svm_classifier.intercept_[0]
                weights[key].append(w)         # w consists of 2 elements
                bias[key].append(b)      # b consists of 1 element
                print(key + " weights: " + str(w))
                print(key + "bias: " + str(b))
                y_pred_calc = np.zeros((178,1))
                for i in range(X_test.shape[0]):
                    y_pred_calc[i] = X_test[i,0]*w[0] + X_test[i,1]*w[1]  + X_test[i,2]*w[2]+ b
                    # if key == "10":
                    #     print("y_pred_calc: " + str(y_pred_calc[i]))
                    if y_pred_calc[i] > 1:
                        y_pred_calc[i] = 1
                    else:
                        y_pred_calc[i] = 0
                accuracy = accuracy_score(y_pred_calc,y_test)
                print(key + " accuracy: " + str(accuracy*100))
                accuracies[key].append(accuracy)
                if accuracy > best_accuracy[key]:
                    best_accuracy[key] = accuracy
                    best_weights[key] = svm_classifier.coef_[0]
                    best_bias[key] = svm_classifier.intercept_[0]
    print("best accuracy: " + str(best_accuracy))
    print("best wieghts: " + str(best_weights))
    print("best bias: " + str(best_bias))


#Sinker values are estimates
experience_levels = {"10" : {"FF" : [54/mean_velo_dict["FF"],1368/mean_spin_dict["FF"]], "FS" : [53/mean_velo_dict["FS"],1000/mean_spin_dict["FS"]], "FC" : [52/mean_velo_dict["FC"],1400/mean_spin_dict["FC"]], "SI" : [53/mean_velo_dict["SI"],1320/mean_spin_dict["SI"]], "SL" : [55/mean_velo_dict["SL"],1255/mean_spin_dict["SL"]], "CH" : [53/mean_velo_dict["CH"],1115/mean_spin_dict["CH"]], "CU" : [59/mean_velo_dict["CU"],1693/mean_spin_dict["CU"]]},
                     "11" : {"FF" : [61/mean_velo_dict["FF"],1514/mean_spin_dict["FF"]], "FS" : [60/mean_velo_dict["FS"],1250/mean_spin_dict["FS"]], "FC" : [59/mean_velo_dict["FC"],1550/mean_spin_dict["FC"]], "SI" : [59/mean_velo_dict["SI"],1500/mean_spin_dict["SI"]], "SL" : [62/mean_velo_dict["SL"],1832/mean_spin_dict["SL"]], "CH" : [57/mean_velo_dict["CH"],1296/mean_spin_dict["CH"]], "CU" : [59/mean_velo_dict["CU"],1701/mean_spin_dict["CU"]]},
                     "12" : {"FF" : [63/mean_velo_dict["FF"],1588/mean_spin_dict["FF"]], "FS" : [62/mean_velo_dict["FS"],1220/mean_spin_dict["FS"]], "FC" : [61/mean_velo_dict["FC"],1600/mean_spin_dict["FC"]], "SI" : [60/mean_velo_dict["SI"],1580/mean_spin_dict["SI"]], "SL" : [60/mean_velo_dict["SL"],1695/mean_spin_dict["SL"]], "CH" : [57/mean_velo_dict["CH"],1287/mean_spin_dict["CH"]], "CU" : [56/mean_velo_dict["CU"],1540/mean_spin_dict["CU"]]},
                     "13" : {"FF" : [67/mean_velo_dict["FF"],1674/mean_spin_dict["FF"]], "FS" : [66/mean_velo_dict["FS"],1212/mean_spin_dict["FS"]], "FC" : [64/mean_velo_dict["FC"],1695/mean_spin_dict["FC"]], "SI" : [65/mean_velo_dict["SI"],1650/mean_spin_dict["SI"]], "SL" : [61/mean_velo_dict["SL"],1707/mean_spin_dict["SL"]], "CH" : [61/mean_velo_dict["CH"],1367/mean_spin_dict["CH"]], "CU" : [58/mean_velo_dict["CU"],1663/mean_spin_dict["CU"]]},
                     "14" : {"FF" : [72/mean_velo_dict["FF"],1795/mean_spin_dict["FF"]], "FS" : [67/mean_velo_dict["FS"],1163/mean_spin_dict["FS"]], "FC" : [68/mean_velo_dict["FC"],1861/mean_spin_dict["FC"]], "SI" : [70/mean_velo_dict["SI"],1750/mean_spin_dict["SI"]], "SL" : [66/mean_velo_dict["SL"],1820/mean_spin_dict["SL"]], "CH" : [66/mean_velo_dict["CH"],1479/mean_spin_dict["CH"]], "CU" : [62/mean_velo_dict["CU"],1775/mean_spin_dict["CU"]]},
                     "15" : {"FF" : [76/mean_velo_dict["FF"],1893/mean_spin_dict["FF"]], "FS" : [69/mean_velo_dict["FS"],1205/mean_spin_dict["FS"]], "FC" : [71/mean_velo_dict["FC"],1883/mean_spin_dict["FC"]], "SI" : [75/mean_velo_dict["SI"],1870/mean_spin_dict["SI"]], "SL" : [68/mean_velo_dict["SL"],1893/mean_spin_dict["SL"]], "CH" : [70/mean_velo_dict["CH"],1553/mean_spin_dict["CH"]], "CU" : [65/mean_velo_dict["CU"],1863/mean_spin_dict["CU"]]},
                     "16" : {"FF" : [79/mean_velo_dict["FF"],1954/mean_spin_dict["FF"]], "FS" : [71/mean_velo_dict["FS"],1160/mean_spin_dict["FS"]], "FC" : [73/mean_velo_dict["FC"],1927/mean_spin_dict["FC"]], "SI" : [77/mean_velo_dict["SI"],1920/mean_spin_dict["SI"]], "SL" : [71/mean_velo_dict["SL"],1981/mean_spin_dict["SL"]], "CH" : [72/mean_velo_dict["CH"],1607/mean_spin_dict["CH"]], "CU" : [68/mean_velo_dict["CU"],1939/mean_spin_dict["CU"]]},
                     "17" : {"FF" : [81/mean_velo_dict["FF"],2011/mean_spin_dict["FF"]], "FS" : [73/mean_velo_dict["FS"],1205/mean_spin_dict["FS"]], "FC" : [76/mean_velo_dict["FC"],2017/mean_spin_dict["FC"]], "SI" : [79/mean_velo_dict["SI"],1980/mean_spin_dict["SI"]], "SL" : [73/mean_velo_dict["SL"],2030/mean_spin_dict["SL"]], "CH" : [75/mean_velo_dict["CH"],1647/mean_spin_dict["CH"]], "CU" : [70/mean_velo_dict["CU"],2001/mean_spin_dict["CU"]]},
                     "18" : {"FF" : [84/mean_velo_dict["FF"],2049/mean_spin_dict["FF"]], "FS" : [76/mean_velo_dict["FS"],1295/mean_spin_dict["FS"]], "FC" : [79/mean_velo_dict["FC"],2075/mean_spin_dict["FC"]], "SI" : [82/mean_velo_dict["SI"],2000/mean_spin_dict["SI"]], "SL" : [76/mean_velo_dict["SL"],2078/mean_spin_dict["SL"]], "CH" : [77/mean_velo_dict["CH"],1678/mean_spin_dict["CH"]], "CU" : [72/mean_velo_dict["CU"],2048/mean_spin_dict["CU"]]},
                     "College" : {"FF" : [85/mean_velo_dict["FF"],2055/mean_spin_dict["FF"]], "FS" : [77/mean_velo_dict["FS"],1221/mean_spin_dict["FS"]], "FC" : [79/mean_velo_dict["FC"],2073/mean_spin_dict["FC"]], "SI" : [83/mean_velo_dict["SI"],2020/mean_spin_dict["SI"]], "SL" : [76/mean_velo_dict["SL"],2086/mean_spin_dict["SL"]], "CH" : [78/mean_velo_dict["CH"],1667/mean_spin_dict["CH"]], "CU" : [73/mean_velo_dict["CU"],2056/mean_spin_dict["CU"]]},
                     "MLB" : {"FF" : [1,1], "FS" : [1,1], "FC" : [1,1], "SI" : [1,1], "SL" : [1,1], "CH" : [1,1], "CU" : [1,1]}   
}

print("experience_levels: " + str(experience_levels))
# train_experience_level(X,y,experience_levels)
train_multiple_iterations(X,y,experience_levels,1)