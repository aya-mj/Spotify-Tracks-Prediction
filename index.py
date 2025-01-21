import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load and clean the dataset
df = pd.read_csv('dataset.csv')
df = df.drop(columns=['Unnamed: 0', 'track_id'])

# Streamlit App
st.title('Spotify Dataset Explorer')
st.markdown("""
This app allows you to:
- Explore and analyze Spotify track data.
- Visualize data distributions and correlations.
- Filter tracks by genre and other features.
- Train a basic machine learning model to predict track popularity.
""")

st.sidebar.header('More Options')

# Data Exploration
st.header('Data Exploration')
st.write(df.head(5))

if st.sidebar.checkbox('Show Summary Statistics'):
    st.subheader('Summary Statistics')
    desc = df.describe()
    st.write(desc)

st.sidebar.subheader('Correlation Heatmap')
if st.sidebar.checkbox('Show Heatmap'):
    numeric_df = df.select_dtypes(include=np.number)  
    corr_matrix = numeric_df.corr() 
    fig2 = plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Feature Correlations')
    st.pyplot(fig2)

# Distribution of a Feature
st.subheader('Distribution of a Feature')
feature = st.selectbox('Choose a feature to plot:', df.select_dtypes(include=np.number).columns)
fig1 = plt.figure(figsize=(10, 6))
sns.histplot(df[feature], bins=50, kde=True, color='blue')
plt.title(f'Distribution of {feature}')
plt.xlabel(feature.capitalize())
plt.ylabel('Count')
st.pyplot(fig1)

# Scatter Plot
st.subheader('Scatter Plot')
x_feature = st.selectbox('Choose X-axis feature:', df.select_dtypes(include=np.number).columns)
y_feature = st.selectbox('Choose Y-axis feature:', df.select_dtypes(include=np.number).columns)
fig3 = plt.figure(figsize=(10, 6))
sns.scatterplot(x=df[x_feature], y=df[y_feature], s=10, alpha=0.7)
plt.title(f'{y_feature} vs {x_feature}')
plt.xlabel(x_feature.capitalize())
plt.ylabel(y_feature.capitalize())
st.pyplot(fig3)

# Most Popular Artists 
st.subheader('Most Popular Artists and Their Songs')
top_n = st.slider('Select Number of Top Artists to Display:', min_value=5, max_value=20, value=10)
popular_artists = (
    df.groupby(['artists', 'track_name'])['popularity']
    .max()
    .reset_index()
    .sort_values(by='popularity', ascending=False)
    .head(top_n)
)
st.write(f'Top {top_n} Most Popular Artists and Their Songs:')
st.dataframe(popular_artists)

# # Plot the data
# fig4 = plt.figure(figsize=(12, 6))
# sns.barplot(
#     x='popularity',
#     y='artists',
#     data=popular_artists,
#     hue='track_name',
#     dodge=False,
#     palette='viridis'
# )
# plt.title(f'Top {top_n} Most Popular Artists and Their Songs')
# plt.xlabel('Popularity')
# plt.ylabel('Artists')
# plt.legend(title='Track Name', bbox_to_anchor=(1.05, 1), loc='upper left')
# st.pyplot(fig4)

st.subheader('Song Filter by Genre')
selected_genre = st.selectbox('Select Genre:', df['track_genre'].unique())
filtered_data = df[df['track_genre'] == selected_genre]
st.write(f'Data filtered by genre: {selected_genre}')
st.write(filtered_data.head(10))

st.sidebar.subheader('Filter Explicit Songs')
show_explicit = st.sidebar.checkbox('Show Only Explicit Songs')
if show_explicit:
    explicit_songs = df[df['explicit'] == True]
    st.write('Displaying Explicit Songs:')
    st.write(explicit_songs) 

st.header('Machine Learning: Predicting Track Popularity')

label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('popularity', axis=1)
y = df['popularity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader('ML Model Performance (Linear Regression)')
st.write(f'Mean Squared Error (MSE): {mse:.2f}')
st.write(f'R-squared (R²): {r2:.2f}')

fig5 = plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, s=10, alpha=0.7)
plt.title('Actual vs Predicted Popularity')
plt.xlabel('Actual Popularity')
plt.ylabel('Predicted Popularity')
plt.axline([0, 0], [1, 1], color='red', linestyle='--', linewidth=1.5, label='Ideal Prediction')
plt.legend()
st.pyplot(fig5)


