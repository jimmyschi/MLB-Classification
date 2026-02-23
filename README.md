Radar Front End:

The received RF signal was downconverted to an Intermediate Frequency (IF) band by
using a radar transceiver system. This implementation was chosen due to its availability,
low cost, and proven success in similar commercial systems. Analog IF signals were
returned from the radar unit in the low mV range, requiring amplification and filtering in
order to condition the signals properly for analog to digital conversion and processing.
Given the system specifications for a 24.125 GHz transmit signal and the velocity range
for detection, the pass band of the filter was defined by the anticipated returning doppler
shift. For signals 10-110 MPH, an approximately 700-8000 Hz passband would
theoretically be designed. However, the designed was overspecced for a 500-10kHz
passband to accommodate non-idealities in the filter design and allow for lower
frequency test signals.

In this configuration the single horn antenna was used for both receiving and
transmitting the radar signal. The Mono Doppler Transceiver, MACOM part number
MACS-007801-0M1RM0, was used as the central component for both the
transmission and reception of radar pulses. The MACS-007801-0M1RM0 Block
Diagram is shown below:

<img width="479" height="206" alt="image" src="https://github.com/user-attachments/assets/a8a49510-920c-47d6-a548-edde919db1a3" />

The 3 pins provide a 5.5V Max DC Input, a Ground Reference, and the IF Output
from the transceiver respectively. The Transceiver operates on less than 1 watt of
power with a minimum outputt power of 5.0 mW. The minimum transceiver
sensitivity is -93 dBc with respect to the transmit power [8].
The Gunn Oscillator operates as the local oscillator, generating the 24.125 GHz for
both transmission and down-conversion, when given a sufficient 5V DC Bias with
reference to ground. 24.125 GHz was chosen due to the readily available RF
Transceiver and the balance this frequency provides between noticeable doppler
effect without sacrificing detectable range. The doppler shift at this frequency is
approximately 161 Hz per 1 m/s of linear velocity (about 72 Hz per MPH). The
modulation index induced by the spinning baseball is large, since the approximately
36mm radius of a baseball is larger than the about 12mm wavelength of the
transmitted wave. The large index of modulation increased the modulated signal
bandwidth, allowing for easier observation of multiple harmonics, but it gave a lower
margin for noise floor due to the smaller peaks in the bessel function side spectra.
This signal is coupled to the horn antenna through a waveguide. The single horn
antenna functions as both a transmitter and receiver of the radar signal. The received
signal is mixed with the transmit signal through a schottkey diode mixer, and the
output is provided via the IF Output pin.

Amplifier/Filter Design
The final Amplifier/Filter implementation consisted of two gain stages with AC
Coupling combined with a low-pass output filter. The passband region was designed
from 500 Hz to 10 KHz. There is a double zero at DC with a corner frequency at 500
Hz. This was designed both to AC Couple the input signal to the first gain stage and in
order to increase the low-frequency noise attenuation. Low frequency noise was of
particular concern due to ambient effects, such as power supply interference, that
could lead to corrupting readings. The high-frequency rolloff was only designed as
first-order for simplicity. Noise in the 10-100 KHz range is far less common, thus
rapid attenuation of these frequencies was not considered a priority.
The Amplifier Gain of the system was estimated and improved upon experimentally
based on observed IF signals from the transceiver output. The overall passband has a
theoretical gain of approximately 4545x, or 73.1 dB. This gain was divided into two
stages. The first stage has a gain of approximately 303x or 49.6 dB. The second stage
has a gain of approximately 15x or 23.5 dB.
The OP27 and OP37 Operational Amplifiers were chosen as the central components
due to their availability, high Gain-Bandwidth-Product, and low noise characteristics.
The OP37 has the higher GBW, exceeding 60 MHz, and thus was chosen for the
higher gain stage (303x). The OP27 was chosen for the second gain stage as it has a
better noise voltage, and the smaller 8 MHz GBW was acceptable for the lower gain
stage (15x). The purpose of orienting the amplifier stages by decreasing gain was to
decrease noise cascading throughout the system. This was also the purpose of only
using two op-amps for gain and filtering, rather than introducing additional active
filters. Minimizing the number of active components reduced the theoretical noise
floor of the front end.
An inverting amplifier configuration was chosen for the first stage of the amplifier
chain as the 330 Ω resistance was an appropriate value for both the high pass response
and gain response of the first stage. An inverting amplifier is also a more stable
configuration due to the isolation between output and ground. A non-inverting
configuration was chosen for the second amplifier stage for the opposite reason. By
removing the dependence between the high-pass filter from the amplifier gain, the
gain could be tweaked and adjusted more readily by modifying the feedback
resistance. This was an important consideration given the rapid prototyping timeline of
this project.
Special attention was given to the selection of passive components towards the front
end of the amplifier chain. Given the voltage scale of the input from the transceiver
minor C1 and C2, minor voltage fluctuations from low grade ceramic capacitors due
to DC bias, piezoelectric, or microphonic effects could amplify and distort the output
signal. Film capacitors are a popular choice to overcome this problem in analogous
high quality audio applications. Thus, metalized film capacitors were chosen as the
most suitable, readily available alternative.

The low-pass filter on the output of the second stage was designed to minimize
attenuation in the passband while providing some rolloff for higher, undesired
frequencies. The 100 Ω resistor was chosen as it provided optimal balance between
passband attenuation and biasing of the protective Zener diode. A .15uF capacitor was
also readily available to best approximate the desired 10KHz cutoff frequency.
Micro-Millivolt range noise potentially produced by low grade ceramic capacitors was
consider of less concern outside of the amplification stage. Thus the smaller form
factor of a ceramic capacitor was deemed the better designed choice in this application
compared to the bulkier metalized film.

<img width="513" height="294" alt="image" src="https://github.com/user-attachments/assets/645378d1-aa1b-4b3b-a357-da763b748eea" />

<img width="488" height="286" alt="image" src="https://github.com/user-attachments/assets/b8ee233f-296f-45c2-9279-137de8d63677" />


Due to timeline constraints and various redesigns, a fully realized PCB for this front
end was not implemented. However, in order to maintain signal integrity in a rapid
prototyping environment Perfboard and solder traces were used to construct the
hardware. The Perfboard design emphasized modularity and compact form factor.
Separate, small boards were constructed for the power supply, amplifier/filter, and the
final signal conditioning. This allowed for easier debugging, design modification, and
flexible orientation inside the enclosure. Solder traces were used whenever possible to
form stable, consistent connections on each perfboard. Jumper wires were used to
interface between different perfboards. While these wires are not immune from
crosstalk, this design choice allowed for flexible prototyping and rapid adjustments.
Ground connections were made directly from the source, using a design principle
commonly referred to as “Star Ground”. By avoiding cascading ground connections
18
through direct routes back to a common ground node, non-ideal currents will not be
allowed to flow through ground loops back into the system. This principle was
intended to add stability to the system, particularly in the first gain stage. For
example, if non-ideal ground currents were to leak into the positive terminal of the
OP37, this would create a positive feedback loop and generate a saturated oscillator.
Using a Star Ground prevents this problem from occurring. Similar design decisions
were applied to all voltage supplies, with direct connections from the source.

Design Implementation
The main task of the embedded module was to accurately sample the incoming
analog signal and perform the necessary calculations to reveal the velocity and spin
rate of a pitch. As mentioned the hardware used to accomplish this was a Teensy 4.1.
At each hardware and software iteration, rigorous testing was performed to evaluate if
that design was suitable for our purposes.

<img width="244" height="292" alt="image" src="https://github.com/user-attachments/assets/7af31e7f-614b-44c5-b230-d2b79dfe1dac" />

The FFT was performed with the arduinoFFT library which provided a structure for
sampling the signal and finding the major peak. Other libraries were tested for this
purpose but did not provide the granularity of sampling required for the purposes
described. It was then possible to perform the doppler shift calculations, which
included parsing the stored samples for the necessary spectral lines to calculate the spin rate. With this process in place to run the calculations when a signal was
received on an analog pin, it was necessary to evaluate the noise in the environment
and include thresholds to limit detections to the area of concern, minimize noise, and
control what was transmitted to the bluetooth module.
The final step in the embedded module was to transmit the velocity and spin rate
calculations to the bluetooth module for classification and display in the application.
This was accomplished using the Arduino hardware serial functions.

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

Backend Hardware:

After much research on different demos other engineers had done
online, the conclusion we came to was to use the HC-05 Bluetooth Module because of
the great quantity documentation for it. This specific Bluetooth module connects to the
serial ports, both TX and RX, of the microcontroller used, and transmits the data
output from those ports via Bluetooth. This also allowed us to use an MCU with much
more memory, rather than the Arduino Nano 33 BLE from the original design.

Backend Software:

In the backend of the app, a custom BluetoothHandler class was made to handle the
process of requesting the correct permissions to allow Bluetooth processes to
complete, searching for the correct device, as well as reading messages from the
HC-05.
MVVM (Model-View-Viewmodel) architecture was used to efficiently connect the
back-end code, including as classes for pitches, pitchers, Bluetooth, and pitch
identification, to the UI elements of the app.
5.1.4.1.6 UI Design
Much of the UI was completed with respect to the conceptual design, with the same
features as expected, with some being in different areas or pages in the application.
The final UI design looks as follows:

<img width="248" height="537" alt="image" src="https://github.com/user-attachments/assets/2409c286-3cdd-4261-a181-86446edd6817" />

with data being collected in the graphs for a specific pitcher, and the most recent pitch
data being displayed in large text. The user also has the ability to add a new pitcher to
start adding pitches to their pitch list and classify their pitches.
