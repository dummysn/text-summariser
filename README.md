# Save and Load Tokenizer

it is important to save the tokenizer used during training for text summarization. The tokenizer plays a crucial role in converting text data into sequences that can be fed into the trained model for generating summaries.

You can save the tokenizer to a file after fitting it on the training data using the save() method. Here's an example:

tokenizer.save('tokenizer.pkl')

Later, when you want to use the trained model to generate summaries, you can load the tokenizer from the saved file using the load() method:

from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
tokenizer.load('tokenizer.pkl')


# Save Text Length

it is recommended to store the max_text_len and max_summary_len values along with the tokenizer when training a text summarization model. These values represent the maximum lengths of the text and summary sequences used during training and are required for preprocessing new text data before generating summaries.

import json

# Save tokenizer and sequence lengths to a JSON file
config = {
    'tokenizer_file': 'tokenizer.pkl',
    'max_text_len': max_text_len,
    'max_summary_len': max_summary_len
}

with open('config.json', 'w') as f:
    json.dump(config, f)

When loading the model for text summarization, you can also load the tokenizer and the configuration file to retrieve the max_text_len and max_summary_len values:

import json
from tensorflow.keras.preprocessing.text import Tokenizer

# Load tokenizer from file
tokenizer = Tokenizer()
tokenizer.load('tokenizer.pkl')

# Load configuration file
with open('config.json', 'r') as f:
    config = json.load(f)

max_text_len = config['max_text_len']
max_summary_len = config['max_summary_len']

# Embedding Dimesion and Layers
However, if you have a relatively smaller dataset or limited computational resources, you may consider reducing the number of layers to improve efficiency and prevent overfitting. You could try values such as 6 or 8 layers instead of the default 12.

If you have limited computational resources or a smaller dataset, you may consider reducing the hidden_units value to improve efficiency and prevent overfitting. You could try values such as 512 or 768.

