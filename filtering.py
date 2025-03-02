import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split


data = {
    "user_id": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5],
    "book_id": [1, 2, 1, 3, 2, 4, 3, 5, 4, 5],
    "rating": [5, 4, 5, 3, 4, 5, 3, 4, 5, 5]
}
df = pd.DataFrame(data)


user_ids = df["user_id"].unique()
book_ids = df["book_id"].unique()

user_mapping = {id: idx for idx, id in enumerate(user_ids)}
book_mapping = {id: idx for idx, id in enumerate(book_ids)}

df["user_id"] = df["user_id"].map(user_mapping)
df["book_id"] = df["book_id"].map(book_mapping)


X = df[["user_id", "book_id"]]
y = df["rating"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


user_input = Input(shape=(1,))
book_input = Input(shape=(1,))

user_embedding = Embedding(input_dim=len(user_mapping), output_dim=10)(user_input)
book_embedding = Embedding(input_dim=len(book_mapping), output_dim=10)(book_input)

user_vec = Flatten()(user_embedding)
book_vec = Flatten()(book_embedding)

concat = Concatenate()([user_vec, book_vec])
dense1 = Dense(64, activation='relu')(concat)
dense2 = Dense(32, activation='relu')(dense1)
output = Dense(1, activation='linear')(dense2)

model = Model([user_input, book_input], output)
model.compile(optimizer='adam', loss='mse')


model.fit([X_train["user_id"], X_train["book_id"]], y_train, epochs=150, batch_size=32)


def recommend_books_for_user(user_id, n=3):
    user_idx = user_mapping[user_id]
    book_indices = np.array(list(book_mapping.values()))
    user_inputs = np.full_like(book_indices, user_idx)

    predictions = model.predict([user_inputs, book_indices])
    top_indices = np.argsort(predictions.flatten())[::-1][:n]

    recommended_book_ids = [list(book_mapping.keys())[i] for i in top_indices]
    return df[df['book_id'].isin(recommended_book_ids)][['book_id']].drop_duplicates()


print(recommend_books_for_user(1))