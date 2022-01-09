## Independent Research Repository 

* Wrote bootstrap ensemble neural network in Python and trained it on the real word E-mini S&P 500 options data to predict the implied volatility. 
Used the predicted implied volatility as an input in the Bjerksund-Stensland 2002 model and achieved more than 99% R Squared for both Call and Put Options. 
It also has decent performances regarding the moneyness of the options and in terms of the mean absolute error or the mean absolute percentage Error. 

The research consists of two more future steps.
* The proposed phase two is to implement reinforcement learning to develop and improve the pricing model for a single agent who uses the trained ensemble neural net to price the options contracts.
* The proposed phase three is to use the ensemble method, with multiple agents in a committee, to find a more robust and powerful strategy in trading the options contract in the simulated market.
