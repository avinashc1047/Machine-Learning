# Machine-Learning
CISC-484 Machine Learning projects in Python using various classification models to solve different tasks.


## Assignment 1
### Feature Extraction from Twitter Data
This project involves building a feature set for a machine learning model used for analyzing Twitter data. The task was to extract at least eight new features from Twitter JSON data and saving them.

- Added eight new features to enhance the dataset, including text-based features
- Extracted detailed information from raw Twitter JSON files, processing both tweet content and user-related metadata.
- Evaluated the impact of additional features on the model's performance by testing various machine learning algorithms, leading to a more robust model.
- Included a comprehensive report detailing the new features, their extraction methods, and a summary of how they were incorporated into the model.

*Technologies Used: Python, Pandas, Scikit-Learn, JSON, Jupyter Notebooks*

## Assignment 2
### Comparing Classification Algorithms for Twitter Spam Detection
This project focuses on comparing the performance of multiple machine learning algorithms for spam classification in Twitter data. It builds on the previous Twitter dataset and compares the effectiveness of four classification algorithms: Support Vector Machine (SVM), Logistic Regression, Random Forest, and Decision Tree. The objective is to assess and analyze the accuracy, precision, and recall of these models on the Twitter spam dataset.

- Implemented four classification algorithms—SVM, Logistic Regression, Random Forest, and Decision Tree—on the Twitter spam dataset.
- Evaluated and compared the performance of these algorithms based on key metrics: Accuracy, Precision, and Recall.
- Delivered a comprehensive report comparing the performance differences between the algorithms and discussed potential reasons behind those differences, such as algorithm complexity, overfitting, and model interpretability.

*Technologies Used: Python, Numpy, Scikit-Learn, Pickle, Jupyter Notebooks*

## Assignment 3
### Custom Implementation of Random Forest for Twitter Spam Classification
In this project, a custom implementation of the Random Forest algorithm was built from scratch to classify spam tweets in the Twitter dataset. The model utilizes decision trees built using information gain and includes measures to avoid overfitting. The project compares the performance of the custom Random Forest implementation with other machine learning models previously tested in earlier assignments.

- Developed a Random Forest algorithm by manually building decision trees using information gain as the splitting criterion.
- Incorporated techniques such as limiting tree depth, using random subsets of features, and bootstrapping to prevent overfitting and improve generalization.
- Provided a detailed report explaining the algorithm implementation, the techniques used to reduce overfitting, and an analysis of the model's performance compared to other algorithms.

*Technologies Used: Python, Numpy, Jupyter Notebooks*

## Assignment 4
### Enhancing Deep Learning Model with Data Augmentation and Network Optimization
This project focuses on improving a modified VGG16 neural network for image classification by applying advanced data augmentation techniques, adding extra layers to the architecture, experimenting with different learning rates and optimizers, and visualizing the training process. The goal is to enhance the model's performance and address issues like overfitting and underfitting.

- Implemented additional data augmentation techniques (e.g., rotation, zoom) to increase dataset diversity and improve model generalization.
- Added layers to the ModifiedVGG16 architecture, making the network deeper to capture more complex features.
- Tested two different optimizers (excluding SGD) and adjusted the learning rate to find the best combination for faster convergence and better performance.
- Plotted accuracy-loss graphs to visualize training progress, and analyzed overfitting and underfitting by observing the model's behavior on both the training and validation sets.

*Technologies Used: Python, Pandas, Numpy, Torch, Matplotlib, Jupyter Notebooks*

## Assignment 5
### Enhancing LSTM Network for Sequence Prediction with Deeper Architecture and Regularization
This project extends the previous LSTM-based model by deepening the network architecture and improving regularization to enhance performance for sequence prediction tasks. The LSTM model is now made deeper with additional layers, and dropout layers are introduced for regularization. The goal is to improve the model's ability to generalize to unseen data and prevent overfitting, while comparing the performance with different optimizers.

- Added additional LSTM layers, increasing the total to four LSTM layers for more complex feature extraction.
- Integrated Dropout layers after LSTM layers and between dense layers to prevent overfitting and improve model robustness.
- Introduced additional dense layers (50 neurons) with ReLU activation to add non-linearity and improve the model’s expressive capacity.
- Tested two new optimizers and compared their impact on model performance and convergence speed.
- Plotted training and testing MAE (Mean Absolute Error) and MSE (Mean Squared Error) to assess model accuracy and error over time.

*Technologies Used: Python, Keras, TensorFlow, Matplotlib, Numpy, Pandas, Sklearn, Jupyter Notebooks*

## Final
### Weather Parameter Prediction Using Machine Learning and LSTM
This project focuses on developing two models for weather parameter prediction using the Mesonet dataset. The goal is to predict future weather conditions (temperature, humidity, and pressure) based on historical data. The project includes two approaches: one using a traditional machine learning algorithm and the other using an LSTM model to predict weather conditions at different time scales.

- Implemented a traditional Random Forest model to predict weather parameters for the next hour using the past 6 hours of weather data.
- Built an LSTM model to predict weather conditions for the next 3 hours based on the past 12 hours of data, leveraging LSTM’s ability to capture temporal dependencies in time-series data.
- Carefully preprocessed the dataset for both models, including handling missing values, normalizing data, and reshaping the data for use in the LSTM model.
- Measured the performance of both models by calculating Mean Squared Error (MSE) for the predictions, providing insights into their accuracy and reliability.

*Technologies Used: Python, Pandas, Numpy, Scikit-learn, Keras, TensorFlow, Matplotlib, Jupyter Notebooks*