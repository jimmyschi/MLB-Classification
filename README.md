Classification Module:

Pitch classification was determined using information sent from the Bluetooth module including spin rate and velocity and a Machine Learning module that is pre trained using previous data collected from the MLB season will classify it. The classification module has two options for classifying, including a two pitch classification that does Fastball vs Offspeed and a seven pitch classification that classifies a 4-Seam-Fastball, Cutter, Sinker,Slider, Splitter, Changeup, and Curveball. The ML model used for training was a linear-kernel Support Vector Machine classifier in created in Python. Key features from this model were then extracted to be used for making predictions on the backend in C#. Finally, I trained separate SVM models for different age groups using a separate dataset that included information on spin rate and velocity for different pitches for age groups from 10 all the way to the pro level. 	

Previous Design Alternatives:

K-Means Clustering:
Through researching various different pitch classification models, K-Means clustering seemed very promising. However, due to the fact that this is an unsupervised ML model it only groups the pitches based on similarity and doesn’t use the correct classification label which was accessible from the 2022 MLB dataset. When pretraining this model on the given dataset there was extremely low accuracies recorded like 25% for both two pitch and seven pitch classification. With results like this it was known that an unsupervised ML model is not ideal for this application and a supervised ML model such as SVM would be a much better alternative.
ML Model on Microcontroller
The original idea was to extract the pretrained ML model from Python into Arduio code that can be compiled on the Microcontroller used in this design. However, during our first attempts at integrating the classification with the embedded module we experienced a very high runtime that caused the system to be unusable. With the limited computing power and memory that the Microcontroller is capable of using, we then pivoted to running the classification on the backend. Running the classification on the backend solved all of these issues and allowed classification to perform more smoothly when integrating the entire system.

Implementation:
Support Vector Machine Two Pitch Classification

Figure 1: SVM Decision Boundary for 2-Pitch Classification
![SVM](https://github.com/user-attachments/assets/f5b8f362-5dd0-44e3-85c1-aee630fc8ba2)

A linear Kernel SVM does the classification for two classes using a weights vector and a bias feature. The weights vector assigns a corresponding value to the importance of each of the different features within the feature vector, therefore, it has the same size as the number of features and the bias feature represents the intercept of the linear decision boundary. Classification will be done by determining if that pitch is above(Fastball) or below(Offspeed) the decision boundary. The accuracy I was able to obtain from splitting the MLB dataset into a training and testing set was 96% using Offspeed pitches as Curveball,Changeup,Slider,Splitter, and Fastball pitches as 4-Seam-Fastball, Cutter, and Sinker. This accuracy is very high and is well above our minimum threshold for accuracy, therefore, our radar gun is able to classify two pitch types very easily at the Major League level. 

Figure 2: Two Pitch Confusion Matrix using 178 Testing Samples
![5799FDC1-58EE-476A-8345-1090B7BF714B](https://github.com/user-attachments/assets/3373c986-9576-4fa7-88d5-894f2b382abf)

Experience based Two Pitch Classification
Experience based Two Pitch Classification performs exactly how Two Pitch Classification at the MLB level works, except there was a separate linear-kernel SVM trained on each of the different experience levels including ages 10-18 and College. The separate dataset that I used contained information on average spin rate and velocity per experience level was obtained from Rapsodo. I used this data to feature scale the entire original MLB dataset for every single pitch for every different experience level. The accuracy for this type of classification ranges from 70% to 89%. The lower end of the range corresponded to the younger age groups because with less experience pitching these younger pitchers were not throwing the specific pitch types correctly yet. Accuracy did increase as age increased and the highest accuracy corresponded to the College level where pitchers were throwing the corresponding pitches more accurately and similar to the Major League level. 
Seven Pitch Classification 
Seven Pitch classification is performed using a one vs one approach for the linear-kernel SVM. Using a OVO approach with seven classes creates 21 different weights vectors and bias features because we need a decision boundary to compare each class to each of the other different classes. Predictions are made the same way as the two pitch classification except we will be doing it 21 times and voting for which class it belongs to. The pitch will then be classified as the class with the most votes. The accuracy for this classification is lower at 70% which still meets our requirements for pitch classification accuracy. The accuracy is lower than the two pitch classification because there are more classes to predict and the model does a poor job in distinguishing between certain pitch types like the Sinker and 4-Seam Fastball due to their similarities in velocity and spin rate which were the main metrics used to classify. 

Figure 3: Seven Pitch Confusion Matrix using 178 Testing Samples
![8CBB5FBE-B342-4737-BEB4-97864F81FCC4](https://github.com/user-attachments/assets/3eb71fad-e28c-4cfa-850b-83176a04ea77)

Experience based Seven Classification Algorithm:

Experience based Seven Pitch Classification works similar to the experience based two pitch classification except it also uses the properties of the seven pitch classification to make the predictions. The accuracies were much lower for this type of classification ranging from 53-65%. The lower percentages also correspond to the younger age groups and accuracy increases as age increases with college having the highest accuracy at 65%. 

Classification on Backend:

The SVMs were pre trained in Python due to accessibility to SVM libraries, ease of use, and familiarity. The weights and bias features were then extracted from each of the pretrained SVMs and prediction functions were created in C# so the classification can be done without any processing or memory limitations. This also made it very easy to receive information like throwing arm and experience level selected from the User Interface and information like velocity and spin rate because they were already being sent to the backend via the Bluetooth module. The classification on the backend performed exactly as the classification did in Python when testing 100 random sample pitches to confirm the new prediction functions.  
