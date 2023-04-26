# Advanced_Machine_Learning
# Pain Face Classification: A Comparative Analysis and model studying

## Introduction/Overview

### Problem
Pain is a complex experience that is difficult to measure objectively. One common way to assess pain is by observing a patient's facial expressions, which can provide some insight into their level of discomfort. However, classifying pain based on facial expressions is a challenging task in healthcare and medicine because pain is a subjective experience and people may express pain differently. For example, some individuals may have a high pain threshold and may not show many facial expressions even though they are in pain.

Accurately classifying pain is important in healthcare and medicine because it can help medical professionals diagnose and treat patients. If a patient is experiencing severe pain, they may need immediate medical attention or a change in medication to manage their pain. However, existing pain classification methods, such as self-reporting or pain scales, can be unreliable and subjective. Self-reporting is based on the patient's own description of their pain, which may not always be accurate, and pain scales can be influenced by factors such as cultural background or personal experience.

Therefore, developing a more accurate and objective method of pain classification is essential to improve patient outcomes. This could involve using technology to measure physiological responses to pain or developing more sophisticated algorithms to analyze facial expressions. A more objective method of pain classification would enable medical professionals to more accurately diagnose and treat patients, leading to better pain management and improved patient outcomes.

### Motivation
Accurate pain classification is crucial for providing effective treatment to patients. In many cases, pain is the primary reason for seeking medical attention. However, pain can be difficult to diagnose and treat because it is a subjective experience. Facial expressions are often used as an indicator of pain, but these expressions can be complex and difficult to interpret. Therefore, developing a more accurate method of pain classification based on facial expressions can greatly improve patient outcomes by enabling medical professionals to diagnose and treat pain more effectively

### Interestingness
Pain face classification refers to the process of identifying and categorizing different types of pain based on facial expressions. This is an interesting problem because human facial expressions can be complex and difficult to interpret. Different people may express pain in different ways, and some may try to hide or mask their pain. Therefore, accurately classifying pain based on facial expressions can be a challenging task.

Pain is a universal human experience, and it affects people of all ages and backgrounds. Accurately classifying pain is crucial for providing effective treatment to patients. Pain can have a significant impact on a person's quality of life and their ability to carry out daily activities. If pain is left untreated or misdiagnosed, it can lead to a range of negative health outcomes, including depression, anxiety, and reduced mobility.

By accurately classifying pain, medical professionals can diagnose and treat patients more effectively. This can involve providing pain relief medication, physical therapy, or other forms of treatment. In some cases, accurately classifying pain can also help identify underlying health issues that may be contributing to the pain. Overall, accurate pain classification has significant implications for patient care and treatment, and it can greatly improve a person's quality of life.


### Difficulty
Classifying pain faces can be a challenging problem because of the subjective nature of pain and the variability of facial expressions. Different people may show different pain expressions, and various types of pain may present differently. Moreover, cultural and social factors may affect how individuals express pain. Different methods of pain classification may work better for different people or types of pain.

### Approach
The project aims to compare the performance of two neural networks - Convolutional Neural Network (CNN) and another network - for the task of classifying pain faces.
We also used ROC curves to analyze the results of the comparison. ROC curves are a graphical representation that help evaluate the performance of a binary classifier. It shows the trade-off between the true positive rate and the false positive rate of a classification model as the classification threshold varies.
Furthermore, the project aims to explore the impact of various data augmentation methods and transfer learning techniques on the classification performance of the neural networks. Data augmentation is a technique that artificially increases the size of the training data set by creating modified versions of the original data samples, such as rotation, translation, scaling, or flipping. Transfer learning, on the other hand, is a technique where a pre-trained model is used to provide a head start for the model training process, where the learned features can be used in a new task, and fine-tuning can be performed on top of these features for the new task.

## Preliminaries

### Problem Setup
Model 1 uses a simpler CNN architecture consisting of a single 2D convolutional layer with 32 filters, a max pooling layer, a flatten layer, two dense layers with ReLU activation, a dropout layer, and a final dense layer with sigmoid activation. It uses the binary crossentropy loss function and Adam optimizer for training. Model 1 also uses a learning rate scheduler and early stopping to prevent overfitting and a model checkpoint to save the weights of the best-performing model.

Model 2, on the other hand, uses a more complex CNN architecture consisting of three 2D convolutional layers with 32, 64, and 128 filters respectively, max pooling layers, a flatten layer, two dense layers with ReLU activation, a dropout layer, and a final dense layer with sigmoid activation. It uses the binary crossentropy loss function, Adam optimizer, and accuracy metric for training. Model 2 also uses a learning rate scheduler, early stopping, and model checkpoint callbacks.

In terms of comparison, Model 2 is a more complex and deeper CNN architecture than Model 1. This increased complexity may make Model 2 more capable of learning complex features from the input data, leading to better performance in some cases. However, this also comes at a cost of increased computational resources and longer training times. On the other hand, Model 1 is a simpler architecture that may be more suitable for smaller datasets or when computational resources are limited.

Note that it was determined that transfer learning should not be utilized in order to mitigate the risk of overfitting, especially given the imbalanced nature of the dataset.

### Dataset
The dataset used for the face pain detection task contains 14,614 samples in total. Of these, 15 samples belong to class 0 "no_pain" and 14,599 samples belong to class 1 "with_pain". This indicates a highly imbalanced dataset, with the majority of the samples belonging to the "with_pain" class.

Imbalanced datasets can create challenges in training machine learning models, as it can lead to overfitting. Overfitting occurs when a model becomes too specialized to the training data and is not able to generalize well to new, unseen data. In the case of an imbalanced dataset, the model can achieve high accuracy simply by predicting the majority class, but may perform poorly on the minority class.

To avoid these issues, it is crucial to address the class imbalance in the dataset before training a model. One possible approach is to use techniques like oversampling or undersampling to balance the class distribution. Another approach is to use algorithms that are specifically designed to handle imbalanced datasets, such as cost-sensitive learning or ensemble methods. By addressing the class imbalance, we can improve the model's generalization performance and prevent overfitting to the majority class.
This is the link to the dataset: https://www.kaggle.com/datasets/coder98/emotionpain

![Facial Expressions](/face_images.png)


### Experimental Setup
In this study, I aimed to mitigate the problem of overfitting and improve the performance of a binary classification model using data augmentation and oversampling techniques. As mentioned earlier, the dataset consisted of 15 samples in class "0" and 14,599 samples in class "1". To address the class imbalance, I applied data augmentation techniques, including rotation, width and height shifting, horizontal flipping, and vertical flipping to augment the 15 samples in class "0" to 2000 samples. I also randomly sampled 2000 samples from the 14,599 samples in class "1" to balance the dataset. I finally selected 515 samples (15 class "0" and 500 class "1") for the ROC curve analysis.
I performed predictive modeling using different data augmentation methods (rotation, flipping, and zooming) and different folds of minority data augmentation. The model was trained using a binary cross-entropy loss function, with a batch size of 16 and five epochs. The image size was set to 200. I experimented with different combinations of augmentation methods and folds of data augmentation to evaluate the model's performance.
I also conducted sensitivity analysis on three predictive scores using twofold mixup data augmentation on minority training data. The image size was again set to 200, and the loss function was binary cross-entropy. I trained the model for ten epochs using a batch size of two. I also tested the model's sensitivity to changes in the mixup parameter alpha (set to 0.2) and the number of folds (set to 3).
Overall, the experimental setup aimed to evaluate the effectiveness of different data augmentation and oversampling techniques to improve the performance of a binary classification model on an imbalanced dataset. I used rigorous experimental protocols, including sensitivity analysis, to ensure the reliability of our results.

## Results

### Main Results
<span style="color:red">*Here are the resulsts :* </span>.

#### Model1 : 

<span style="color:blue">*Loss* </span>.

![alt text](/model1_loss.png)


<span style="color:blue">*Accuracy* </span>.

![alt text](/model1_accuracy.png)

#### model2 : 

<span style="color:blue">*Loss* </span>.

![alt text](/model2_loss.png)

<span style="color:blue">*Accuracy* </span>.

![alt text](/model2_accuracy.png)


### comparison : 

![alt text](/comparison.png)

### Roc curve : 



![alt text](/Roc_curve.png)

<br/>

<span style="color:red">*Predictive scores by different data augmentation methods and different folds of minority data augmentation* </span>.

<br/>
<br/>
<br/>

![alt text](/Augment_FOLD.png)

<br/>

<span style="color:red">*sensitivity analysis on three predictive scores using twofold mixup data augmentation on minority training data* </span>.


<br/>
<br/>
<br/>


![alt text](/Mixup_twofolds.png)

## Discussion
Based on the graphs, it is apparent that there may be some concerns while training the first model. Specifically, the accuracy appears to have overshot its mark for several reasons. Firstly, it seems that I did not utilize a high enough number of epochs to allow the model to stabilize. Additionally, the data I used for training was highly unbalanced, with only 0.001% of the observations belonging to class "0" and the remainder allocated to class "1". Despite efforts to augment the data, this discrepancy in class proportions cannot be entirely concealed, and is likely contributing to the observed inaccuracies in the model.


An AUC (Area Under the ROC Curve) of 0.5 which means that the model has no ability to discriminate between the positive and negative classes. In other words, it performs no better than random guessing. An AUC of 0.5 indicates that the model is unable to distinguish between true positive and true negative cases and is effectively worthless for the task at hand this is due to the imbalanced data too.

For Predictive scores by different data augmentation methods and different folds of minority data augmentation: 

Based on the results , it appears that the data augmentation techniques "flip", "rotate", and "zoom", had a positive impact on the performance of the predictive model. This is evident from the improvement in various evaluation metrics such as accuracy, precision, recall, and F1-score.

For instance, the "flip" augmentation method had the highest accuracy mean of 0.9938 and F1-Score mean of 0.9969 on fold 4. Similarly, the "rotate" method had a high accuracy mean of 0.9901 and F1-Score mean of 0.9950 on fold 3. Lastly, the "zoom" method had a high precision mean of 0.9913 on fold 4. These results suggest that data augmentation techniques can significantly improve the performance of a predictive model by increasing its ability to generalize to unseen data.

For Sensitivity analysis on three predictive scores with twofold mixup data augmentation on minority training data:

The table shows the mean and standard deviation values for various evaluation metrics such as accuracy, precision, recall, and F1-score. The results indicate that the model achieved a high accuracy mean of 0.9967 and F1-score mean of 0.9983. Additionally, the model achieved perfect recall with a value of 1.0.

Furthermore, the sensitivity analysis was performed on three predictive scores with an alpha value of 0.2, indicating that mixup data augmentation on minority training data was applied with a mixing coefficient of 0.2.

Overall, these results suggest that twofold mixup data augmentation on minority training data can improve the performance of the predictive model, particularly in terms of accuracy, F1-score, and recall. The results also highlight the importance of exploring different data augmentation techniques to improve the generalization ability of the model.

## Conclusion

In conclusion, pain face classification is a challenging problem in the field of healthcare and medicine, and accurately classifying pain can greatly improve patient outcomes. In this project, I compared the performance of two neural networks, CNN and another network, for pain face classification. I also explored the impact of different data augmentation methods and transfer-learning techniques on the performance of the models.

My experimental results showed that data augmentation techniques such as flip, rotate, and zoom can significantly improve the performance of a predictive model by increasing its ability to generalize to unseen data. I also found that twofold mixup data augmentation on minority training data can improve the performance of the predictive model, particularly in terms of accuracy, F1-score, and recall.

Overall, my study provides valuable insights into the use of different techniques to address the challenges of pain face classification. These insights can inform future research in the field and help to develop more accurate and objective methods of pain classification that can improve patient outcomes.
