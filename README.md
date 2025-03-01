# Stock_Prediction_LSTM_CUDA




**Project Background**
This project focuses on predicting Nike's stock prices using Long Short-Term Memory (LSTM) neural networks. The data was gathered from historical stock market records, primarily using Yahoo Finance. Initially, I preprocessed the data by normalizing the closing prices and structuring it into time-series sequences. The goal was to build a model that could capture trends and patterns in stock movements. Through multiple iterations, I experimented with different batch sizes, sequence lengths, dropout rates, and optimizers to improve the model’s accuracy. The final version leverages an optimized LSTM architecture with dropout layers for regularization, achieving stable predictions over time.


**Executive Summary**
I worked on developing an LSTM-based model to predict Nike's stock prices using historical closing price data. The data was gathered and preprocessed, ensuring it was clean and normalized for better training performance. A sequence length of 60 days was used, meaning the model predicts the next day's price based on the last 60 days of data.

**Model Development and Training**
The model architecture consisted of three LSTM layers, with Batch Normalization and Dropout applied after each layer to prevent overfitting. I experimented with different batch sizes (32, 64, and 128) and found that smaller batch sizes resulted in lower loss values. The model was trained using the AdamW optimizer, with early stopping implemented to halt training if validation loss stopped improving. Learning rate adjustments were handled by a scheduler to ensure smooth convergence.

**Performance and Observations**
Loss values varied significantly based on batch size. A batch size of 32 achieved the lowest loss (0.00006), while 128 resulted in 0.0042. Additionally, I tested increasing the sequence length from 60 to 120 days, but the model's performance declined, suggesting it struggled with capturing longer-term dependencies effectively. The model followed the overall stock price trends but showed excessive volatility in predictions, indicating potential overfitting.
![Screenshot 2025-03-01 135815](https://github.com/user-attachments/assets/ee30a2e6-3f18-4a22-8e18-2fd78c7ad155)


**Future Improvements**
To stabilize results, I focused on optimizing dropout placement, ensuring dropout layers were applied after each LSTM layer instead of within them to maintain cuDNN optimizations. I also fine-tuned hyperparameters, such as batch size and learning rate, to find the best configuration. Further improvements could include testing different loss functions, experimenting with GRU layers, and incorporating additional market indicators like trading volume and moving averages to enhance predictive power.

Before hyperparamater tuning: 
 ![Screenshot 2025-03-01 111454](https://github.com/user-attachments/assets/be10e791-d458-422f-ac19-e5eb2fa39579)
![Screenshot 2025-03-01 111521](https://github.com/user-attachments/assets/f48d72b1-93f6-4121-ab15-845f034a574f)



After hyperparamater tuning:
![Screenshot 2025-03-01 135830](https://github.com/user-attachments/assets/9ee19495-77fb-46de-9e94-d53b1a45aeef)


**Challenges I Faced along the way:**

**CUDA Setup Problems:**

Issue: I encountered problems setting up CUDA, which is crucial for utilizing the GPU to accelerate deep learning computations. It was initially challenging to ensure compatibility between the versions of CUDA, cuDNN, and the TensorFlow version I was using.
Solution: After troubleshooting, I confirmed that the appropriate versions of CUDA and cuDNN were installed (CUDA-- 11.5 , CUDNN-- 8.3.1 Tensorflow-- 2.10, Python-- 3.10) . I then made sure TensorFlow could access the GPU by running tests that verified proper GPU usage. Adjusting the environment paths and ensuring compatibility with the system's GPU helped resolve the issue.
Why I Did It: Using the GPU was critical for training the LSTM model efficiently, especially with large datasets like stock prices. CUDA-enabled GPUs significantly accelerated the training process, reducing computational time and making model experimentation more feasible.

Infinity Numbers and NaN Predictions:

Issue: During training, the model produced infinity (Inf) and NaN (Not a Number) values, which is a common problem in deep learning models. These values can result from unstable gradients or improper data handling, such as learning rates that are too high or issues with data scaling.
Solution: I applied several strategies to address this:
Gradient Clipping: This technique was used to prevent gradients from exploding during backpropagation, which could cause Inf or NaN values.
Data Normalization: I scaled the stock price data properly, ensuring that the values fell within a range that made training stable.
Lowered Learning Rate: I adjusted the learning rate to avoid overly large updates to the model’s weights, which could lead to unstable training and NaN values.
Regularization: I added dropout layers to prevent overfitting, which can sometimes cause unpredictable behavior like NaNs in predictions.
Why I Did It: Ensuring the model could train stably was crucial for getting reliable predictions. Infinity and NaN values not only affect the performance of the model but also disrupt the training process. Stabilizing the training with proper techniques helped the model converge successfully.

**Project Deep Dive**

Data Preprocessing and Scaling:

I began by loading the stock price data from a CSV file into a DataFrame. I then converted the 'date' column to a datetime format and set it as the index of the DataFrame. The dataset was filtered to include only the 'close' price column, which is the key feature for my prediction task. This step was crucial because it ensured that the dataset was organized with time-series information that could be leveraged for sequence-based learning with LSTM.

To ensure stable training, I applied MinMax scaling to the 'close' column. The scaling was done within the range of (0.001, 1) to avoid zero values, as LSTMs can be sensitive to extremely small or large values. I then clipped the scaled values to ensure that they stayed within this range, preventing potential instability. The scaled data contributed to the model by providing it with normalized values, making it easier for the LSTM to learn without being influenced by outliers

After scaling, I created sequences of length 90 from the closing price data. These sequences represented 90 days of stock price history, and each sequence would be used to predict the next day's price. This sequence setup allowed the LSTM to learn temporal dependencies and trends over a set period, which is critical for stock price prediction.

Data Splitting:

Once the data was prepared, I split it into training, validation, and test sets. The training set contained 70% of the data, the validation set had 10%, and the test set had the remaining 20%. This step was necessary to evaluate the model’s performance on unseen data and ensure that it could generalize well. The training set helped the model learn the underlying patterns, the validation set was used to tune hyperparameters and prevent overfitting, and the test set was kept aside to provide a final evaluation.

Model Architecture:

I created an LSTM model using TensorFlow/Keras. The model consisted of three LSTM layers, with 128, 64, and 64 units, respectively. These layers were stacked sequentially, with each having 'tanh' activation and 'sigmoid' for the recurrent activation. LSTM layers are well-suited for time series data like stock prices because they capture temporal dependencies. The first two layers were set with return_sequences=True, meaning that each of those layers would output sequences for the next layer to process, while the final LSTM layer had return_sequences=False to output a single value (the prediction).

Dropout layers were added after each LSTM layer to prevent overfitting, a common problem in deep learning. Dropout randomly disables a fraction of neurons during training, making the model less likely to overfit to the training data. I also added batch normalization after each LSTM and dropout layer. This helped stabilize and accelerate training by normalizing the activations in each layer, improving convergence speed.

The final layer was a Dense layer with a linear activation function, which is suitable for regression tasks like predicting stock prices. This structure helped the model focus on learning from the sequential nature of stock prices, while also preventing overfitting and ensuring smooth training.

Model Compilation and Optimization:

I compiled the model with the AdamW optimizer, which is a variant of Adam with weight decay. This optimizer provides better generalization and helps reduce overfitting. I set a learning rate of 0.00005 and a weight decay of 1e-6 to balance the model’s performance and prevent large weight updates. I also applied gradient clipping using clipnorm=0.5 to prevent exploding gradients, which could lead to NaN values and instability in training.

The loss function I used was Mean Squared Error (MSE), which is common for regression tasks and penalizes larger prediction errors more heavily, encouraging the model to focus on minimizing these errors during training.

Callbacks and Learning Rate Scheduling:

To enhance the model’s performance, I implemented two key callbacks: early stopping and learning rate scheduling. Early stopping monitored the validation loss, and if the loss did not improve after 20 epochs, it would stop the training and restore the best weights. This prevented unnecessary training and ensured that the model didn’t overfit to the training data.

I also used a learning rate schedule with exponential decay, starting with a learning rate of 1e-5. This learning rate decayed by 10% every 500 steps, which helped the model converge more effectively over time. Gradually lowering the learning rate is known to help in achieving better model accuracy in the later stages of training, ensuring fine-tuned predictions.

Model Training:

I trained the model for 100 epochs with a batch size of 32. The training was done using the training data (X_train and Y_train) and validated using the validation set (X_valid and Y_valid). Training for a sufficient number of epochs allowed the model to learn intricate patterns in the stock prices, while the validation set helped ensure that the model didn’t overfit and that it could generalize well.

Each step I took in this process—from data preprocessing and scaling to model design and training—contributed to making the model more effective in learning from historical stock prices and producing accurate predictions for future stock prices. The LSTM model, with its ability to capture temporal dependencies, was ideal for this task. The careful consideration of overfitting, the use of gradient clipping, dropout, and batch normalization ensured stable and efficient training, and the optimizer and learning rate schedule helped the model converge to the best solution.

This approach provided me with valuable insights into how LSTMs can be used in stock price prediction and allowed me to develop a model that could provide meaningful predictions, helping to inform investment decisions.
