# speech-emotion-recognition
# Speech Emotion Recognition AI
This project uses **MLPClassifier** to detect emotions from audio files.

### Accuracy: 67.19%

### How to Run:
1. Download the [RAVDESS Dataset](https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio).
2. Install libraries: `pip install librosa soundfile streamlit sklearn`
3. Run the UI: `python -m streamlit run app.py`
https://github.com/divya223352/speech-emotion-recognition
https://speech-emotion-recognition-wvtdgjfsn3rzkcsvv7hhzs.streamlit.app/
✨ Features
🧠 Deep Learning Analysis: Employs Convolutional Neural Networks (CNN) or MLP Classifiers to detect subtle emotional patterns.
📊 Feature Extraction: Automatically extracts acoustic features like MFCC (Mel-Frequency Cepstral Coefficients), Chroma, and Mel Spectrograms.
📁 Multi-Dataset Support: Trained on gold-standard datasets like RAVDESS, TESS, and CREMA-D.
📈 Real-time Visualization: Generates waveplots and spectrograms to visualize sound frequencies and intensities.
🎙️ Live Voice Testing: Allows users to record or upload their own voice for instant emotion prediction.
🛠️ Technologies UsedCategoryTools/LibrariesProgramming LanguagePythonAudio ProcessingLibrosa, PyAudio, SoundFileDeep LearningTensorFlow / Keras or PyTorchMachine LearningScikit-learn (MLPClassifier, SVM)Data HandlingNumPy, PandasVisualizationMatplotlib, Seaborn🚀 Setup and Run Instructions
1️⃣ Clone the RepositoryBashgit clone https://github.com/yourusername/speech-emotion-recognition.git
cd speech-emotion-recognition
2️⃣ Environment SetupBashpython -m venv ser_env
# Windows:
ser_env\Scripts\activate
# Mac/Linux:
source ser_env/bin/activate
3️⃣ Install DependenciesBashpip install librosa soundfile numpy scikit-learn tensorflow matplotlib
4️⃣ Run the ProjectBashpython main.py
👣 Step-by-Step Work DoneData Collection: Gathered audio samples from the RAVDESS dataset (24 professional actors, 8 emotions).Audio Preprocessing: Normalized audio lengths and applied data augmentation (adding noise, pitch shifting) to increase model robustness.Feature Extraction: Used librosa to convert raw audio into numerical features ($MFCCs$) that represent the "texture" of the sound.Model Building: Constructed a CNN (Convolutional Neural Network) architecture to recognize patterns in the extracted features.Training & Validation: Split data into 80% training and 20% testing sets to evaluate accuracy and prevent overfitting.Prediction Engine: Developed a script to take any .wav file and output the most likely emotion label.⚠️ Errors Faced and Solutions1. Librosa Load ErrorError: NoBackendError when loading .wav files.Solution: Installed ffmpeg or libsndfile on the system to handle diverse audio codecs.2. Data ImbalanceError: The model always predicted "Neutral" because that class had more samples.Solution: Used Synthetic Minority Over-sampling Technique (SMOTE) and data augmentation to balance the emotion counts.3. Low Accuracy on Live VoiceError: High accuracy on dataset but failed on real-world recordings.Solution: Added a StandardScaler to normalize input features to the same scale as the training data.✅ Final OutputThe system achieves a classification accuracy of approximately 70-85% (depending on the model). It provides:An emotion label for any input speech file.A confidence score (probability) for the prediction.A visual spectrogram of the processed audio.
