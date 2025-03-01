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


