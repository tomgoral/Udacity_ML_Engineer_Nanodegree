# Machine Learning Degree Capstone Project

This repository contains code and associated files for deploying a product cost predictor using AWS SageMaker.

## Project Overview

As a product manager for a major fastener manufacturer, I am caught between timeliness and accuracy of cost proposals. On the one hand, a timely cost estimate will please a customer. On the other hand, an inaccurate cost estimate can frustrate a customer or backfire on the fastener manufacturer if the cost is too low! A method to provide timely and accurate economic proposals is needed if our company wants to grow with this competitive market.



This project will be broken down into three files:

**Notebook 1:  Anonymizer**

* Reads in profitability data .
* Extracts for machine learning.
* Obfuscates confidential labels with generic ones.
* Outputs the public domain version data set "data.xlsx"

**Notebook 2: Feature Engineering**

* Clean and pre-process data.
* Select "good" features, by analyzing the correlations between different features.
* Create train/validate/test csv file

**Notebook 3: Train - Deploy Model**

* Upload train/validate/test feature data to S3.
* Examine candidate ML models .
* Train the model and deploy it using SageMaker.
* Evaluate the deployed predictor model.

