
# Prediction of S&P500 index value

### Introduction

* The Standard and Poor's 500, or simply the S&P 500, is a stock market index tracking the stock performance of 500 large companies listed on exchanges in the United States. It is one of the most commonly followed equity indices.
* In this project we have used machine learning to predict its value on a given date. We have used machine learning models LSTM or Long Short Term Memory for making predictions.




![App Screenshot](https://upload.wikimedia.org/wikipedia/commons/thumb/7/7e/S_and_P_500_chart_1950_to_2016_with_averages.png/330px-S_and_P_500_chart_1950_to_2016_with_averages.png)

### Dataset
* We have S&P500 data from june 2012 to june 2022.It has 2548 rows. It has several columns like opening price , closing price, low and high. For the training of the model we have used closing price of a day.

### Data Analysis 
* For technical Analysis, the focus has been on the most prominent indicators that can be efficiently operationalized and are intuitive in interpretation, including: Moving average convergence & divergence; Stochastic KD; Relative Strength index; Larry William's R%; Daily Closing Volume. 
* For Economic analysis, indicators being utilized in terms of their importance and data availability are: Gross Domestic Product; Consumer price Index; Producer Price Index; Employment Index; Fed Fund Rate.
* Technical indicators have been calculated from the downloaded daily closing prices & volume data. The closing price & volume have been smoothend using Welles Wilder Smoothing without any look-ahead bias, and relative 15days change is calculated to serve as the price & volume trend indicators. 
* Economical Indicators have been extracted from the officially released historical percentage change data. In addition, Fed Funds rate have been smoothend using Welles Wilder Smoothing and relative 15 day change has been applied to vapture the trends in the economic cycle.


#### Technical Analysis
* **1. Moving average convergence & divergence** :- It is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price
* MACD Formula:-
**MACD=12-Period EMA − 26-Period EMA**

* ![Moving_Average](https://user-images.githubusercontent.com/92113558/186102928-8a7e8b7a-96ec-4f16-b294-b5118e517529.png)


*  If the MACD crosses above its signal line following a brief correction within a longer-term uptrend, it qualifies as **bullish** confirmation.
*  If the MACD crosses below its signal line following a brief move higher within a longer-term downtrend, traders would consider that a **bearish** confirmation.

* **2. Market Trend** :- 
* A trend is the overall direction of a market or an asset's price. In technical analysis, trends are identified by trendlines or price action that highlight when the price is making higher swing highs and higher swing lows for an uptrend, or lower swing lows and lower swing highs for a downtrend.
* An uptrend is marked by an overall increase in price. Nothing moves straight up for long, so there will always be oscillations, but the overall direction needs to be higher in order for it to be considered an uptrend. Recent swing lows should be above prior swing lows, and the same goes for swing highs. Once this structure starts to break down, the uptrend could be losing steam or reversing into a downtrend. Downtrends are composed of lower swing lows and lower swing highs.
* ![Market_trend](https://user-images.githubusercontent.com/92113558/186103724-1209d181-a4b8-45b7-8aa3-1bdcc37f847e.png)

* **3. RSI & CCI**:- 
* **The Relative Strength Index (RSI)**
    * The RSI compares the relationship between the average of up-closes versus the average of down-closes over specific time intervals, usually 14 days. 
    * Values produced by its formula are then plotted on a moving line underneath the price chart. 
    * All readings oscillate between zero and 100, with a midpoint of 50, allowing for easy readings about potential overbought (above 70) and oversold (below 30) levels.
    * ![RSI](https://user-images.githubusercontent.com/92113558/186104434-d82dbeff-1a4c-42c0-9862-41b932665cad.png)


* **The Commodity Channel Index (CCI)**
    * Originally developed to spot cyclical trends in commodities, the CCI has become popular in equities and currencies as well. 
    * The CCI's formula compares an asset's typical price to its moving average and then divides those by the absolute value of its mean deviation from the typical price. 
    * High positive readings signal that the asset is trading more strongly than its past trend cycles predict that it should. 
    * Low negative readings suggest that it is trading weakly. Unlike the RSI, the CCI does not have specific range bounds, which can make it more difficult to read.
    * ![CCI](https://user-images.githubusercontent.com/92113558/186104498-8c591a92-9727-4cce-9c63-4d21d0d8544d.png)


* **4. Williams %R**:-
* It is also known as the Williams Percent Range, is a type of momentum indicator that moves between 0 and -100 and measures overbought and oversold levels. 
* The Williams %R may be used to find entry and exit points in the market. 
* The indicator is very similar to the Stochastic oscillator and is used in the same way. 
* It was developed by Larry Williams and it compares a stock’s closing price to the high-low range over a specific period, typically 14 days or periods.
* ![Williams%R](https://user-images.githubusercontent.com/92113558/186105050-5df5da4c-a6ef-4032-ba26-1548f2529935.png)
* Williams %R moves between zero and -100.
* A reading above -20 is overbought.
* A reading below -80 is oversold.

* **5. Stochastic Oscillator**:- 
* A stochastic oscillator is a momentum indicator comparing a particular closing price of a security to a range of its prices over a certain period of time. 
* The sensitivity of the oscillator to market movements is reducible by adjusting that time period or by taking a moving average of the result. 
* It is used to generate overbought and oversold trading signals, utilizing a 0–100 bounded range of values.
* ![Stocastic_Oscilarator](https://user-images.githubusercontent.com/92113558/186105558-e81478e4-9cc0-467a-8b5e-4607521a13f2.png)
* The stochastic oscillator is range-bound, meaning it is always between 0 and 100. 
* This makes it a useful indicator of overbought and oversold conditions. Traditionally, readings over 80 are considered in the overbought range, and readings under 20 are considered oversold.


### Model used:
* For model training LSTM model is used.Long short-term memory (LSTM) belongs to the complex areas of Deep Learning. It is not an easy task to get your head around LSTM. It deals with algorithms that try to mimic the human brain the way it operates and to uncover the underlying relationships in the given sequential data.

* In our project we have used closing price from june 2012 to june 2022 for training.First we split data into train and test with 80:20 ratio.
The closing price of last 150 days is used for fitting to 151th day price.
* ![Model_accuracy_plot](https://user-images.githubusercontent.com/92113558/186106467-88e10bc4-546a-4118-bd7a-f7a3e64bae76.png)

### Steps for deployment of Model using Streamlit
The app is developed using streamlit library in python.

1-First we import the required libraries.

2-Stock details like Current Mkt price, Previous Closing Price, etc. 

3-Technical Analysis using Indicators

4-Then we load the model we have saved using keras.

5-By using the yfinance library we load the previous 150 closing price of index.

6-Then we take the user_input in which date is entered in ("dd/mm/yy") format for which user want to know the index value.

7-Then model predict the value of stock for that day and plot the graph.

#### Requirements
![App Screenshot](https://user-images.githubusercontent.com/56593219/175295481-76829f59-9ccd-477c-921c-4d4a1f4072c9.png)

**User interface**
* ![Screenshot (6)](https://user-images.githubusercontent.com/92113558/186108718-bbd0ece5-a590-46c0-88d8-961cc5d1e323.png)
* ![Screenshot (7)](https://user-images.githubusercontent.com/92113558/186108775-62e8ad45-f9ae-4b36-be1b-18fe20e01445.png)
* ![Screenshot (11)](https://user-images.githubusercontent.com/92113558/186108823-4c944b83-23fb-4746-871e-c59a597983b4.png)



**Browse link:**

* **App link** - [(https://pavanwanjari-ml-stock-price-app-g36tbn.streamlitapp.com/)](https://pavanwanjari-ml-stock-price-app-g36tbn.streamlitapp.com/)

* **Link to github code using streamlit**:





