import re
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

def preprocess_dataset(dataset_path, output_file_path):
  """Preprocesses a dialog dataset for AI chatbot application.

  Args:
    dataset_path: The path to the dataset.txt file.
    output_file_path: The path to the output file.
  """

  # Load the dataset.txt file.
  dataset = pd.read_csv(dataset_path, sep='\t', names=['question', 'answer'])

  # Clean the text.
  dataset['question'] = dataset['question'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))
  dataset['answer'] = dataset['answer'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', x.lower()))

  # Create a tokenizer.
  tokenizer = Tokenizer(num_words=2500)
  tokenizer.fit_on_texts(dataset['question'] + ' ' + dataset['answer'])

  # Encode the questions and answers.
  encoder_inputs = tokenizer.texts_to_sequences(dataset['question'])
  decoder_inputs = tokenizer.texts_to_sequences(dataset['answer'])
  # Pad the sequences to the same length.
  encoder_inputs = pad_sequences(encoder_inputs, maxlen=30, padding='post')
  decoder_inputs = pad_sequences(decoder_inputs, maxlen=30, padding='post')

  # Save the preprocessed dataset.
  with open(output_file_path, 'w') as f:
    for encoder_input, decoder_input in zip(encoder_inputs, decoder_inputs):
      f.write(str(encoder_input) + ' ' + str(decoder_input) + '\n')

if __name__ == '__main__':
  dataset_path = 'dialogs.txt'
  output_file_path = 'preprocessed_dataset.txt'

  preprocess_dataset(dataset_path, output_file_path)
