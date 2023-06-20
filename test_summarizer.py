# Load the trained model
model = tf.keras.models.load_model('best_model.h5')

# Generate a summary for a given text
def generate_summary(text):
    text_seq = tokenizer.texts_to_sequences([text])
    text_seq = pad_sequences(text_seq, padding='post', maxlen=max_text_len)
    predicted = model.predict([text_seq, np.zeros((1, max_summary_len - 1))])
    predicted = np.argmax(predicted, axis=-1)
    summary = ' '.join(tokenizer.index_word[idx] for idx in predicted[0] if idx > 0)
    return summary

# Example usage
text = "The cat sat on the mat. The dog played in the garden. The bird sang a beautiful song."
summary = generate_summary(text)
print("Summary:", summary)
