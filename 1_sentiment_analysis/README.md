# Deploy a Sentiment Analysis Application <br>

Welcome to the SageMaker deployment project! In this project you will construct a recurrent neural network for the purpose of determining the sentiment of a movie review using the IMDB data set. You will create this model using Amazon's SageMaker service. In addition, you will deploy your model and construct a simple web app which will interact with the deployed model.<br> 

The deployment project which you will be working on is intended to be done using Amazon's SageMaker platform. In particular, it is assumed that you have a working notebook instance in which you can clone the deployment repository. Your project will be reviewed by a Udacity reviewer against the deployment project rubric. Review this rubric thoroughly, and self-evaluate your project before submission. All criteria found in the rubric must meet specifications for you to pass. 

## Preparing and Processing Data 

Review_to_words do?, Remove stop words, Stem words to remove suffixees Delete HTML keywords Convert to lowercase<br>
TOP FIVE TOKENS<br>
1: movi   51,695<br>
2: film   48,190<br>
3: one    27,741<br>
4: like   22,799<br>
5: time   16,191<br>

Created a word dictionary<br>

## Build and Train a PyTorch Model 

The train method is implemented and can be used to train the PyTorch model.<br>
The RNN is trained using SageMaker's supported PyTorch functionality. 

## Deploy a Model for Testing 

Deploy the trained model<br>
How does this model compare to the XGBoost model?<br>
The RNN is a deep learning algorithm that uses the actual words and word count that would be more accurate than XGBoost<br>

## Deploying a Web App 

The web app is deployed, The model is deployed and the Lambda / API Gateway integration is complete so that the web app works (make sure to include your modified index.html).<br> 
Give an example review and response, Answer gives a sample review and the resulting predicted sentiment.<br> 

Suggestions to Make Your Project Stand Out! 

(1) MAKE A BETTER WEB APP <br>
The web app that you make in this project simply reports to the user whether the predicted sentiment was positive or negative. Can you think of a better web app that uses the same model?<br> 
(2) IMPROVE THE WEB APP APPEARANCE<br>
The provided web app is very simple and there is plenty of room for improvement if you wish to stretch your web developer skills.<br>
(3) IMPROVE THE MODEL<br>
The model chosen here is a straightforward RNN with a single hidden layer. There are many different model architectures that you could try to see if they improve the results. 
In this project, you will be tasked with building a plagiarism detector that examines a text file and performs binary classification; labeling that file as either *plagiarized* or *not*, depending on how similar that text file is to a provided source text. Detecting plagiarism is an active area of research; the task is non-trivial and the differences between paraphrased answers and original work are often not so obvious.

