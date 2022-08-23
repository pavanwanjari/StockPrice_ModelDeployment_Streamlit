import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import streamlit as st
import plotly.graph_objects as go
import datetime as dt
import yfinance as yf
import pandas_ta as ta
from plotly.subplots import make_subplots
from datetime import timedelta
from datetime import date

start = '2016-01-01'
end = date.today()

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker', '^GSPC')

stock_info = yf.Ticker(user_input).info
# stock_info.keys() for other properties you can explore
company_name = stock_info['shortName']
st.subheader(company_name)
market_price = stock_info['regularMarketPrice']
previous_close_price = stock_info['regularMarketPreviousClose']
st.write('market price : ', market_price)
st.write('previous close price : ', previous_close_price)

df = data.DataReader(user_input, 'yahoo', start, end)

# describing data

st.subheader('Data from 2016-2022')
#df= df.reset_index()
st.write(df.tail(10))
st.write(df.describe())
# Force lowercase (optional)
df.columns = [x.lower() for x in df.columns]

st.subheader('Technical Analysis')
infoType = st.radio(
        "Choose Technical Analysis Type",
        ('Moving Average Chart', 'Market trend', 'RSI & CCI', 'Williams %R', 'Stochastic Oscillator')
    )
if infoType == 'Moving Average Chart':
	st.subheader('Closing Price vs Time Chart with 100 MA')
	ma100 = df.close.rolling(100).mean()
	fig = plt.figure(figsize = (12, 6))
	plt.plot(ma100)
	plt.plot(df.close)
	st.pyplot(fig)

	st.subheader('Closing Price vs Time Chart with 100 MA & 200MA')
	ma100 = df.close.rolling(100).mean()
	ma200 = df.close.rolling(200).mean()
	fig = plt.figure(figsize = (12, 6))
	plt.plot(ma100, 'g')
	plt.plot(ma200, 'r')
	plt.plot(df.close, 'b')
	plt.legend()
	st.pyplot(fig)

elif infoType == 'Stochastic Oscillator':
	st.subheader('Stochastic Oscillator')

	# Calculate the MACD and Signal line indicators
	# Calculate the short term exponential moving average
	ShortEMA = df["close"].ewm(span = 12 , adjust = False).mean()
	# Calculate the long term exponential moving average
	LongEMA = df["close"].ewm(span = 26, adjust = False).mean()
	# Calculate the MACD line
	MACD = ShortEMA - LongEMA
	#Calculate the signal line 
	signal = MACD.ewm(span = 9, adjust = False).mean()
	# Create new columns for the data
	df["MACD"] = MACD
	df["Signal Line"] = signal


	# Find minimum of 14 consecutive values by rolling function
	df['14-low'] = df['low'].rolling(14).min()
	df['14-high'] = df['high'].rolling(14).max()

	# Apply the formula
	df['%K'] = (df['close'] -df['14-low'] )*100/(df['14-high'] -df['14-low'] )
	df['%D'] = df['%K'].rolling(3).mean()

	
	
	# Force lowercase (optional)
	df.columns = [x.lower() for x in df.columns]
	# Construct a 2 x 1 Plotly figure
	fig2 = plt.figure(figsize = (16, 10))
	fig2 = make_subplots(rows=2, cols=1)
	# price Line
	fig2.append_trace(go.Scatter(x=df.index, y=df['open'], line=dict(color='#ff9900', width=1), 
                            name='open',legendgroup='1',), row=1, col=1 )

	# Candlestick chart for pricing
	fig2.append_trace( go.Candlestick(x=df.index, open=df['open'], high=df['high'], low=df['low'], 
        	close=df['close'], increasing_line_color='#ff9900', decreasing_line_color='black', 
                                 showlegend=False ), row=1, col=1)

	# Fast Signal (%k)
	fig2.append_trace(go.Scatter(x=df.index, y=df['%k'], line=dict(color='#ff9900', width=2), name='macd',
	        # showlegend=False,
	        	legendgroup='2',), row=2, col=1)

	# Slow signal (%d)
	fig2.append_trace(go.Scatter(x=df.index, y=df['%d'], line=dict(color='#000000', width=2),
	        # showlegend=False,
	        	legendgroup='2', name='signal'), row=2, col=1)

	# Colorize the histogram values
	colors = np.where(df['macd'] < 0, '#000', '#ff9900')
	# Plot the histogram
	fig2.append_trace(go.Bar(x=df.index, y=df['macd'], name='histogram', marker_color=colors, ), row=2, col=1)

	# Make it pretty
	layout = go.Layout(autosize=False,
    	width=1000,
    	height=1000, plot_bgcolor='#efefef',
    	# Font Families
    	font_family='Monospace',font_color='#000000', font_size=20,
    	xaxis=dict(
	        rangeslider=dict(visible=True) ))

	# Update options and show plot
	fig2.update_layout(layout)
	st.plotly_chart(fig2)



elif infoType == 'RSI & CCI':
	st.subheader('Relative Strength Index (RSI) & Comodity Channel Index (CCI)')

    
	df["RSI(2)"]= ta.rsi(df['close'], length= 2)
	df["RSI(7)"]= ta.rsi(df['close'], length= 7)
	df["RSI(14)"]= ta.rsi(df['close'], length= 14)
	df["CCI(30)"]= ta.cci(close=df['close'],length=30, high= df["high"], low =  df["low"])
	df["CCI(50)"]= ta.cci(close= df['close'],length= 50, high= df["high"], low =  df["low"])
	df["CCI(100)"]= ta.cci(close= df['close'],length= 100, high= df["high"], low =  df["low"])

	fig3=plt.figure(figsize=(15,15))
	ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
	ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
	ax1.plot( df['close'], linewidth = 2.5)
	ax1.set_title('CLOSE PRICE')
	ax2.plot(df['RSI(14)'], color = 'orange', linewidth = 2.5)
	ax2.axhline(30, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.axhline(70, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.set_title('RELATIVE STRENGTH INDEX')
	st.pyplot(fig3)



	fig4= plt.figure(figsize=(15,15))
	ax1 = plt.subplot2grid((10,1), (0,0), rowspan = 4, colspan = 1)
	ax2 = plt.subplot2grid((10,1), (5,0), rowspan = 4, colspan = 1)
	ax1.plot(df['close'], linewidth = 2.5)
	ax1.set_title('CLOSE PRICE')
	ax2.plot(df['CCI(30)'], color = 'orange', linewidth = 2.5)
	ax2.axhline(-100, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.axhline(100, linestyle = '--', linewidth = 1.5, color = 'grey')
	ax2.set_title('COMMODITY CHANNEL INDEX')
	st.pyplot(fig4)

elif infoType == 'Williams %R':
	st.subheader('Williams %R')
	def get_wr(high, low, close, lookback):
    		highh = high.rolling(lookback).max() 
    		lowl = low.rolling(lookback).min()
    		wr = -100 * ((highh - close) / (highh - lowl))
    		return wr

	df['wr_14'] = get_wr(df['high'], df['low'], df['close'], 14)

	fig5= plt.figure(figsize=(15,12))
	ax1 = plt.subplot2grid((11,1), (0,0), rowspan = 5, colspan = 1)
	ax2 = plt.subplot2grid((11,1), (6,0), rowspan = 5, colspan = 1)
	ax1.plot(df['close'], linewidth = 2)
	ax1.set_title('CLOSING PRICE')
	ax2.plot(df['wr_14'], color = 'orange', linewidth = 2)
	ax2.axhline(-20, linewidth = 1.5, linestyle = '--', color = 'grey')
	ax2.axhline(-50, linewidth = 1.5, linestyle = '--', color = 'green')
	ax2.axhline(-80, linewidth = 1.5, linestyle = '--', color = 'grey')
	ax2.set_title('WILLIAMS %R 14')
	st.pyplot(fig5)
else:
        start = dt.datetime.today() - dt.timedelta(2 * 365)
        end = dt.datetime.today()
        #df = yf.download(user_input, start, end)
        df = df.reset_index()
        fig = go.Figure(
            data=go.Scatter(x=df.index, y=df['adj close'])
        )
        fig.update_layout(
            title={
                'text': "Stock Prices Over Past Two Years",
                'y': 0.9,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'})
        st.plotly_chart(fig, use_container_width=True)




st.subheader("Prediction of Stock Price")

# splitting date into training and testing 
data_training= pd.DataFrame(df['close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['close'][int(len(df)*0.70): int(len(df))])

print("training data: ",data_training.shape)
print("testing data: ", data_testing.shape)


# scaling of data using min max scaler (0,1)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)


#Load model 
model = load_model("keras_model.h5")

#testing part
past_100_days = data_training.tail(100)

final_df= past_100_days.append(data_testing, ignore_index =True)

input_data = scaler.fit_transform(final_df)


x_test = []
y_test = []

for i in range (100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)    


y_predicted = model.predict(x_test)

scaler = scaler.scale_

scale_factor = 1/scaler[0]

y_predicted = y_predicted * scale_factor

y_test = y_test* scale_factor


# final Graph
st.subheader("Predictions vs Original")
fig2= plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

   






st.subheader('Stock Price Prediction by Date')

df1=df.reset_index()['close']
scaler=MinMaxScaler(feature_range=(0,1))
df1=scaler.fit_transform(np.array(df1).reshape(-1,1))
#datemax="24/06/2022"
datemax=dt.datetime.strftime(dt.datetime.now() - timedelta(1), "%d/%m/%Y")
datemax =dt.datetime.strptime(datemax,"%d/%m/%Y")
x_input=df1[:].reshape(1,-1)
temp_input=list(x_input)
temp_input=temp_input[0].tolist()


date1 = st.date_input("Enter Date in this format yyyy-mm-dd")

result = st.button("Predict")
#st.write(result)
if result:
	from datetime import datetime
	my_time = datetime.min.time()
	date1 = datetime.combine(date1, my_time)
	#date1=str(date1)
	#date1=dt.datetime.strptime(time_str,"%Y-%m-%d")

	nDay=date1-datemax
	nDay=nDay.days

	date_rng = pd.date_range(start=datemax, end=date1, freq='D')
	date_rng=date_rng[1:date_rng.size]
	lst_output=[]
	n_steps=x_input.shape[1]
	i=0

	while(i<=nDay):
    
	    if(len(temp_input)>n_steps):
        	  #print(temp_input)
        	    x_input=np.array(temp_input[1:]) 
        	    print("{} day input {}".format(i,x_input))
        	    x_input=x_input.reshape(1,-1)
        	    x_input = x_input.reshape((1, n_steps, 1))
        		#print(x_input)
        	    yhat = model.predict(x_input, verbose=0)
        	    print("{} day output {}".format(i,yhat))
        	    temp_input.extend(yhat[0].tolist())
        	    temp_input=temp_input[1:]
        	    #print(temp_input)
        	    lst_output.extend(yhat.tolist())
        	    i=i+1
	    else:
        	    x_input = x_input.reshape((1, n_steps,1))
        	    yhat = model.predict(x_input, verbose=0)
        	    print(yhat[0])
        	    temp_input.extend(yhat[0].tolist())
        	    print(len(temp_input))
        	    lst_output.extend(yhat.tolist())
        	    i=i+1
	res =scaler.inverse_transform(lst_output)
#output = res[nDay-1]

	output = res[nDay]

	st.write("*Predicted Price for Date :*", date1, "*is*", np.round(output[0], 2))
	st.success('The Price is {}'.format(np.round(output[0], 2)))

	#st.write("predicted price : ",output)

	predictions=res[res.size-nDay:res.size]
	print(predictions.shape)
	predictions=predictions.ravel()
	print(type(predictions))
	print(date_rng)
	print(predictions)
	print(date_rng.shape)

	@st.cache
	def convert_df(df):
   		return df.to_csv().encode('utf-8')
	df = pd.DataFrame(data = date_rng)
	df['Predictions'] = predictions.tolist()
	df.columns =['Date','Price']
	st.write(df)
	csv = convert_df(df)
	st.download_button(
   		"Press to Download",
   		csv,
  		 "file.csv",
   		"text/csv",
  		 key='download-csv'
	)
	#visualization

	fig =plt.figure(figsize=(10,6))
	xpoints = date_rng
	ypoints =predictions
	plt.xticks(rotation = 90)
	plt.plot(xpoints, ypoints)
	st.pyplot(fig)
