import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from bertopic import BERTopic
import plotly.express as px

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

# Topic Modeling with BERTopic
st.subheader("📋 Topic Insights")

# Extract text data for topic modeling
texts = data['text'].fillna('').tolist()

# Load or Train BERTopic Model
try:
    topic_model = BERTopic.load("models/bertopic_model")
    st.success("BERTopic model loaded successfully!")
except FileNotFoundError:
    st.warning("BERTopic model not found. Training model now...")
    topic_model = BERTopic()
    topics, probs = topic_model.fit_transform(texts)
    topic_model.save("models/bertopic_model")
    st.success("BERTopic model trained and saved!")

# Visualizing Topic Distribution
fig = topic_model.visualize_barchart(top_n_topics=10)
st.plotly_chart(fig)

# Display Key Topics
topic_info = topic_model.get_topic_info()
st.dataframe(topic_info)
