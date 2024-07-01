import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from PIL import UnidentifiedImageError

# Custom CSS to change the background color to navy blue
st.markdown(
    """
    <style>
    .reportview-container {
        background: navy;  /* Navy blue background color */
    }
    .stTextInput {
        background-color: white;  /* Set input box background to white */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load the dataset
df = pd.read_csv('/Users/anisnabila/Developer/FYP/FYP2/cleaned_data.csv')

# Label encoding
label_encoder = LabelEncoder()
df['username_encoded'] = label_encoder.fit_transform(df['reviews.username'])
df['product_name_encoded'] = label_encoder.fit_transform(df['product_name'])

# Prepare sequences of item interactions for each user
sequences = df.groupby('username_encoded')['product_name_encoded'].apply(list).values

# Pad sequences to the same length
max_sequence_length = max(len(seq) for seq in sequences)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length, padding='post')

# Train-test split
X_train, X_test = train_test_split(padded_sequences, test_size=0.2, random_state=42)

# Define LSTM Autoencoder model
latent_dim = 10  # Dimensionality of the latent space

input_seq = tf.keras.layers.Input(shape=(max_sequence_length,))
embedding = tf.keras.layers.Embedding(input_dim=len(df['product_name_encoded'].unique()), output_dim=latent_dim, input_length=max_sequence_length)(input_seq)
encoded = tf.keras.layers.LSTM(latent_dim)(embedding)

decoded = tf.keras.layers.RepeatVector(max_sequence_length)(encoded)
decoded = tf.keras.layers.LSTM(latent_dim, return_sequences=True)(decoded)
decoded = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(len(df['product_name_encoded'].unique()), activation='softmax'))(decoded)

autoencoder = tf.keras.models.Model(inputs=input_seq, outputs=decoded)
encoder = tf.keras.models.Model(inputs=input_seq, outputs=encoded)

autoencoder.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Prepare the target data for autoencoder training
X_train_targets = np.expand_dims(X_train, axis=-1)
X_test_targets = np.expand_dims(X_test, axis=-1)

# Train the autoencoder
autoencoder.fit(X_train, X_train_targets, epochs=1, batch_size=128, validation_data=(X_test, X_test_targets))

# Get encoded representations
encoded_users = encoder.predict(padded_sequences)

# Function to get top N recommendations for a specific user ID
def get_top_n_recommendations(user_id, n=10):
    user_representation = encoded_users[user_id]
    all_items = df['product_name_encoded'].unique()
    
    # Extract the correct embedding layer's weights
    item_embeddings = autoencoder.layers[1].get_weights()[0]  # Adjust this index if necessary
    similarity_scores = np.dot(item_embeddings, user_representation)
    
    # Get top N items with highest similarity scores
    top_n_items = np.argsort(similarity_scores)[-n:][::-1]
    return top_n_items, similarity_scores

# Streamlit app
st.title("Products Recommendation System")

# User ID input
user_id_to_recommend = st.number_input("Enter user ID (0-32) to recommend items:", min_value=0, max_value=len(df['username_encoded'].unique())-1)

if user_id_to_recommend in df['username_encoded'].values:
    # Retrieve encoded user ID
    encoded_user_id = user_id_to_recommend

    # Retrieve previous purchases for the user
    previous_purchases = df[df['username_encoded'] == encoded_user_id][['product_name', 'product_image']].drop_duplicates()

    # Get top N recommendations
    top_10_recommendations, similarity_scores = get_top_n_recommendations(encoded_user_id, n=10)
    recommended_product_names = label_encoder.inverse_transform(top_10_recommendations)

    # Display previous purchases and top recommendations
    st.subheader(f"Previous Purchases for User ID {user_id_to_recommend}:")
    for _, row in previous_purchases.iterrows():
        st.write(f"- {row['product_name']}")
        if pd.notna(row['product_image']) and isinstance(row['product_image'], str):
            try:
                st.image(row['product_image'], width=200)
            except UnidentifiedImageError:
                pass

    st.markdown("<h2 style='color: red;'>Highly Recommended Product:</h2>", unsafe_allow_html=True)
    rank = 1
    for product_encoded in top_10_recommendations:
        product_name = label_encoder.inverse_transform([product_encoded])[0]
        product_image = df[df['product_name_encoded'] == product_encoded]['product_image'].iloc[0]
        if pd.notna(product_image) and isinstance(product_image, str):
            st.write(f"{rank}. {product_name}")
            try:
                st.image(product_image, width=200)
                rank += 1
            except UnidentifiedImageError:
                pass
else:
    st.write("Invalid user ID. Please enter a valid user ID.")
