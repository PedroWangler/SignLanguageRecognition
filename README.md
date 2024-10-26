# SignLanguageRecognition

This is an action detection project that uses camera vision to detect signs and display the respective word in real time. The signs are detected using an LSTM model, which is trained by data samples collected using the Data_Collection.py file. 

The MediaPipe holistic framework is used to detect face, hands, and body landmarks for each frame. A single data sample of a sign is stored as a folder, which has files respective to each frame of that sign's video recording. The data inside each file is not an image or any pixel-related data but an array of the landmarks detected in the frame it pertains to.

Once the LSTM model is trained and tested, the run.py file is used to test it in real time and display its predictions. In every frame mediaPipe holistic detects the relevant landmarks, which are then put into an array containing a sequence of landmarks detected in the previous 30 frames. The LSTM model makes a prediction using the array of landmarks from the previous 30 frames, and if the prediction is different than the previous prediction, the predicted word is then displayed on screen.
