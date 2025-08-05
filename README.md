**[Assignment 1: KNN]**  
(1) Implementation of the k-nearest neighbor (k-NN) algorithm by yourself and  
(2) Using the k-NN for classification based on MNIST dataset available at http://yann.lecun.com/exdb/mnist/ or on wine quality available at https://www.kaggle.com/shelvigarg/wine-quality-dataset/  


**[Assignment 2: Regression for housing price prediction]**  
Design different regression models to predict housing prices based on Boston price dataset, which can be downloaded directly on kaggle (https://www.kaggle.com/datasets/vikrishnan/boston-house-prices) or loaded by scikit-learn(check details on https://scikit-learn.org/1.0/modules/generated/sklearn.datasets.load_boston.html).   

You should try at least these models (required):Ridge regression, Lasso regression  


**[Assignment 3: SVM]**  
Based on the MNIST dataset(import using keras.datasets, or available at https://www.kaggle.com/datasets/hojjatk/mnist-dataset/data), use SVM classifier (you can use scikit-learn api) to classify the handwritten digits.  
  
Requirements:  
(1)try different kernels, and find the best hyperparameter settings (including kernel parameters and the regularization parameter) for each of the kernel types  
(2)visualize SVM boundary  
(3)try other methods to classify moist dataset, such as:  
-least squares with regularization  
-Fisher discriminant analysis (with kernels)  
-Perceptron (with kernels)  
-logistic regression  
-MLP-NN with two different error functions.  


**[Assignment 4: MLP]**  
TASK : MNIST Classification By MLP  

HINT: Use torchvision API to get the mnist dataset (some handwriting figures), convert them from images to vector tensors. Then , try to build nn.Linear() layers, (which equals to W & b). Try to feed the vector tensors to Linear Layer and Activation Functions , to get the predict label . Use the loss function to compute the difference between predict label and the ground truth label, use loss.backward() function to get the gradient of each Linear Layer's parameters. Finally, use optimizers (SGD or Others) to make the model converge. 

