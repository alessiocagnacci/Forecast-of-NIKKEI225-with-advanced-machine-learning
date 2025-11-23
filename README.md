# Forecast-of-NIKKEI225-with-advanced-machine-learning
I created and tested both univariate and multivariate LSTM model for NIKKEI225, and then select the best-performing option

I decided to build an advanced machine learning model (LSTM) to forecast the performance of Japan’s main stock market index (Nikkei 225).

The univariate model was optimized by tuning the hyperparameters until achieving a validation loss below 0.001. In the region where the RMSE is minimized, it still remains fairly high—around ¥1200—reflecting the stochastic nature of the market and the absence of exogenous variables.

Next, I attempted to add an exogenous variable—the S&P 500—by creating a multivariate model, which was also optimized to reach a validation loss below 0.001. However, the RMSE for this model remains consistently above ¥1500.

Therefore, adding the S&P 500 only introduced noise and led to a deterioration in performance, RMSE 300¥ above, suggesting that it is not a significant predictor for the Nikkei 225.

As a result, the univariate model is the preferable choice,providing the highest possible accuracy without introducing unnecessary complexity.

I developed the models in Python using the following libraries: numpy, pandas, yfinance, sklearn, tensorflow, matplotlib.


