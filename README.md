# Real Estate Price Prediction using Machine Learning [pdf](https://github.com/johnTrs/SalePricePrediction-XGBoost/blob/main/%CE%A4%CE%B5%CE%BB%CE%B9%CE%BA%CE%B7%CE%91%CE%BD%CE%B1%CF%86%CE%BF%CF%81%CE%B1.pdf)

This project aims to predict the selling price of real estate using a machine learning technique called **Gradient Boosted Trees**. Accurately predicting the price of a property is crucial for future investors, appraisers, and tax assessors. Traditional property price prediction is based on the comparison of cost and selling price, which is not always reliable. Therefore, a prediction model helps to fill in information gaps and improve the efficiency of the real estate market.

## Data Source
The data set used in this project comes from the Ames Housing Dataset, which was compiled by Dean De Cock for use in data science education. This dataset contains information on 1460 real estate properties in Ames, Iowa, USA.

## Methodology

In summary, our approach is based on the machine learning technique called gradient boosting. This technique relies on an ensemble of weak prediction models. Specifically, we used two variations of this technique, relying on decision trees: 
1) Gradient Boosted Trees with mean squared error cost function 
2) XGBoost for decision trees with mean squared error cost function and L2 regularization. 

Finally, we analyzed the significance of the features provided by the models in influencing the sale price and proposed future approaches to improve the results.


## Tools Used
Although we have fully programmed the ML models, we use the following Python libraries for computational speed:
<p >  <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer"> Scikit-learn<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" alt="scikit_learn" width="40" height="40"/> </a> </p>

<p align="left"> <a href="https://pandas.pydata.org/" target="_blank" rel="noreferrer"> Pandas for data manipulation<img src="https://raw.githubusercontent.com/devicons/devicon/2ae2a900d2f041da66e950e4d48052658d850630/icons/pandas/pandas-original.svg" alt="pandas" width="40" height="40"/> </a> <a href="https://scikit-learn.org/" target="_blank" rel="noreferrer">

<a href="https://seaborn.pydata.org/" target="_blank" rel="noreferrer"> Seaborn for data visualization <img src="https://seaborn.pydata.org/_images/logo-mark-lightbg.svg" alt="seaborn" width="40" height="40"/> </a> </p>


## Conclusion
The XGBoost has the best RMSE, with less overfitting on the training data and achieves an RMSE close to the target of 0.1 between the logarithm of the sale price and the predicted price. The overfitting of the models suggests that there may be room for improvement in the prediction, which can be achieved by collecting more data, exploring more models, and using a meta-model based on them. Additionally, the models give us information on the importance of the features in the sale price, so we can use the most important features to make predictions in countries like Greece, where the real estate market differs from the US and not all data may be readily available.
