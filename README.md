# deepLearning - Deep learning case study

## A GUIDE TO THE SCRIPTS

These scripts relate to the analyses reported in the manuscript "Deep Learning in Automated Text Classification: A Case Study Using Toxicological Abstracts"



### The main features of each script are:

#### 1. A machine learning function that:
  - fits a model to the training data;
  - applies the predictive model to the validation data text and the test data text;
  - generates predicted model metrics by compared predicted text labels on the validation data to their actual labels;
  - generates actual performance model metrics by compared predicted text labels on the test data to their actual labels;
  - calls the training-validation visualization function (used internally during hyperparameter turning and not an explicit output);
  - returns the predicted and actual performance metrics.
#### 2. A plotting function that visualizes the training loss and accuracy metrics versus the validation loss and accuracy metrics
#### 3. A Main script that:
  - Splits the available annotated data at random into:
    - 500 data points to be used as validation data in all following analyses;
    - 2500 data points to be used as test data in all following analyses;
    - a training data pool of the remaining 3822 data.
    - (For example, in the script "dlt_bi_lstm_CNN_W2V_v2.py", the data are partitioned in the lines 223 to 235).
  - Implements a loop that:
    - incrementally adds 500 datapoints to the training dataset from the training data pool to help generate data for a learning curve;
    - calls the machine learning function and stores the results.

>Note:  As a shortcut, we rely on global variable declarations within the main machine learning function to access the training, test and validation datasets created in the main script. We should have used arguments instead but as noted these scripts have not been optimized to follow best coding practices.
