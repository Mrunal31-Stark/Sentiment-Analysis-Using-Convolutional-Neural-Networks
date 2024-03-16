### Project Title:
Sentiment Analysis on IMDb Movie Reviews Using LSTM(Long Short -Term Memory)

### Project Overview:
The project aims to perform sentiment analysis on IMDb movie reviews using LSTM (Long Short -Term Memory). Sentiment analysis is a natural language processing (NLP) task that involves classifying text data into positive, negative, or neutral sentiments. In this project, we focus on classifying IMDb movie reviews as either positive or negative sentiments based on the textual content.

### Libraries Used:
- NumPy: For numerical computations and array operations.
- Keras: Deep learning library for building and training neural networks.
- Matplotlib: For data visualization and plotting model performance metrics.
- IMDb Dataset: For accessing the movie reviews dataset.
- textblob :  a pre-trained model for analyzing sentiment polarity (positive, negative, or neutral) of text data. 
### Workflow:
Certainly, let me explain the model you've implemented:

1. **Importing Libraries**: You start by importing necessary libraries such as NumPy for numerical operations, TextBlob for sentiment analysis, and Keras components for building and training the neural network model.

2. **Loading IMDb Dataset**: You load the IMDb movie review dataset using Keras. This dataset contains movie reviews along with their corresponding sentiment labels (positive or negative).

3. **Data Preprocessing**: After loading the dataset, you preprocess it by padding sequences to the same length. This is important because neural networks require input data to have uniform dimensions. You use `sequence.pad_sequences()` from Keras to achieve this.

4. **Defining the Model Architecture**:
   - **Embedding Layer**: This layer converts integer-encoded vocabulary indices into dense vectors of fixed size. It essentially learns to map each word to a high-dimensional vector space where similar words have similar representations.
   - **SpatialDropout1D Layer**: Spatial dropout randomly drops entire 1D feature maps (channels) instead of individual elements. This helps prevent overfitting by adding noise to the input data.
   - **LSTM Layer**: Long Short-Term Memory (LSTM) is a type of recurrent neural network (RNN) architecture capable of learning long-term dependencies. It's well-suited for sequence data like text. In this model, the LSTM layer processes the embedded sequences.
   - **Dense (Output) Layer**: This is the final layer of the neural network, consisting of a single neuron with a sigmoid activation function. It produces a probability output indicating the sentiment (positive or negative) of the input review.

5. **Compiling the Model**: Before training, you compile the model using the binary cross-entropy loss function (suitable for binary classification tasks), the Adam optimizer, and accuracy as the evaluation metric.

6. **Training the Model**: You train the model using the training data (`X_train` and `y_train`) for a specified number of epochs and batch size.

7. **Evaluating the Model**: After training, you evaluate the model's performance on the test data (`X_test` and `y_test`) and print the accuracy.

8. **Sentiment Analysis Example**: Finally, you demonstrate sentiment analysis using TextBlob on the text "It is the worst movie ever." You calculate the sentiment polarity using TextBlob and classify the sentiment as positive, negative, or neutral based on the polarity value.


### Conclusion:
Overall, this model is a basic sentiment analysis model using LSTM for text classification on the IMDb movie review dataset. It learns to predict the sentiment of movie reviews as either positive or negative based on the words present in the reviews.

Through meticulous data preprocessing, model design, training, and evaluation, the project achieves a certain level of accuracy in sentiment classification on the IMDb dataset. The LSTM model's performance provides insights into the predictive power of deep learning models for text classification tasks.



See the file and you can make changes as per the need for your data .
