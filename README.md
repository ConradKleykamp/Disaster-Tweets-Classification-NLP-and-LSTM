# Disaster-Tweets-Classification-NLP-and-LSTM
Using a RNN with LSTM to predict whether or not a Tweet contains language which identifies a disaster

![image](https://github.com/user-attachments/assets/e7a36dd7-6ea9-4b0f-a69c-8daa74100329)

### Objective
This project was completed as part of CU Boulder's Unsupersived Algorithms in Machine Learning course and as part of Kaggle's 'Natural Language Processing with Disaster Tweets' competition. The dataset contains 10,000 Tweets that are already classified. The purpose of this project is to develop a model that correctly predicts which Tweets are about real disasters and which are not. When applied to real life use, a model's ability to correctly identify Tweets about real disasters is important, as certain words and phrases can be used metaphorically or ironically. For example, in the Tweet shown above, the user writes that the sky was "ABLAZE". With the photo of a beautiful sunset provided, a typical person would able to use context clues and determine that this Tweet is not about a real disaster. However, this context is not as clear to a machine learning model.

---

### Methods
Exploratory Data Analysis
- Histogram of the distribution of 'target' classes (Seaborn)
- Histogram of the distribution of # of characters in the Tweets (Seaborn)
- Histogram of the distribution of # of words in the Tweets (Seaborn)

Natural Language Processing
- Removing URLs
- Removing HTML tags
- Removing emojis
- Removing punctuation
- Tokenizing the text (nltk)

Building the Model
- Recurrent Neural Network (RNN) with Long-Short Term Memory (LSTM)
- Total parameters: 2,076,605
- Trainable parameters: 42,305
- Embedding layer --> dropout layer --> LSTM layer --> dense layer (sigmoid activation function)
- Adam optimzer, learning rate = 1e-5

Fitting the Model
- Train (80%), Test (20%)
- Epochs = 15
- Batch size = 4
- Metrics: Loss, Accuracy

---

### General Results
![image](https://github.com/user-attachments/assets/16195152-f0af-4f96-8b3e-9422c11d3b0b)

Best Results (Epoch 15)
- Accuracy: 0.7862
- Validation Accuracy: 0.7997
- Loss: 0.4760
- Validation Loss: 0.4500

From the results above, we can see that the 15th epoch yielded the best results. However, similar performance was experienced around the 10th epoch and then began to have diminishing returns. Due to this, it may not be necessary or worth it to continue to the 15th epoch if one were to factor in the time and cost of computation. In contrast, a GPU could be used here to measure the performance of a larger number of epochs. Furthermore, a smaller batch size could be used to obtain more granular results. This would also likely require the use of a GPU.
