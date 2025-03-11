import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/Tweets.csv")

# Sentiment Analysis
st.title("✈️ Airline Sentiment Analysis")
sentiment_counts = data['airline_sentiment'].value_counts()
fig1, ax1 = plt.subplots()
ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%.2f%%')
st.pyplot(fig1)

# Trend Analysis
st.subheader("📈 Sentiment Trends")
data['tweet_created'] = pd.to_datetime(data['tweet_created'])
trend_data = data.groupby(data['tweet_created'].dt.date)['airline_sentiment'].value_counts().unstack().fillna(0)
st.line_chart(trend_data)

st.subheader("📋 Topic Insights (Add your BERTopic model here)")
st.write("This section will display topic modeling insights once integrated.")