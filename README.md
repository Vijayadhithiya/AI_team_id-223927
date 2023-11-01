# NAALAI THIRAN IBM,
# DOMAIN :Artificial intelligence, 
# PROJECT TITLE : AI CHATBOT using Python,

# Source Data set: dialogs.txt
            
    
# The Required Libraries and Dependencies for Chatbot

    import re,
    import pandas as pd,
    import spacy,
    from flask import Flask, render_template, request,
    from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Steps To run the chatbot :
    1.create a virtual environment using terminal using #python -m venv myenv,
    2.Activate the virtual environment using #myenv/scipts/Activate,
    3.install the required libraries after activating the virtual environmrnt,
    4.In the installed library "FLASK" we need to insert our HTML file as a template in the templates model for web application,
    5.After all the 4 steps completed the chatbot is ready to run for deployment,
    6.To run and deploy the chatbot web application #python chatbot.py .on the terminal,
    7. Now the chatbot will deployed on the webpage , and user can ask the quries to the chatbot and get the response .

# Template:index.html

    <!DOCTYPE html>
    <html lang="en">

    <head>
    <meta charset="UTF-8">
    <meta http-equiv="X-UA-Compatible" content="IE=edge">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chatbot</title>
    <style>
        body {
            display: flex;
            background-color: #007bff;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
        }

        .container {
            
            max-width: 400px;
            width: 100%;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }

        .message {
            margin-bottom: 10px;
        }

        .user-message strong {
            color: #007bff;
        }

        .bot-message strong {
            color: #28a745;
        }
    </style>
    </head>

    <body>
    
    <div class="container">
        <h1 style="text-align: center;">Chatbot</h1>
        <div class="message user-message">
            <strong>You:</strong> {{ user_input }}
        </div>
        <div class="message bot-message">
            <strong>Bot:</strong> {{ bot_response }}
        </div>
        <form method="POST" action="/chat" style="text-align: center;">
            <label for="user_input">You:</label>
            <input type="text" id="user_input" name="user_input" value="{{ user_input }}">
            <input type="submit" value="Ask">
        </form>
    </div>
    </body>
    </html>





# CHATBOT.py
    # Load SpaCy model
    nlp = spacy.load("en_core_web_sm")

    # Flask setup
    app = Flask(__name__)

    # Load the GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Load the dataset from the specified file path
    dataset = pd.read_csv('dialogs.txt', delimiter="\t", header=None, names=["question", "answer"])

    # Define the clean_text function to preprocess text data
    def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.lower()
    return text

    # Define the remove_repeating_sentences function to remove repeating sentences from a dataset
    def remove_repeating_sentences(dataset):
    seen_sentences = set()
    filtered_dataset = []

    for index, row in dataset.iterrows():
        if row["question"] not in seen_sentences:
            seen_sentences.add(row["question"])
            filtered_dataset.append(row)

    return pd.DataFrame(filtered_dataset)

    # Preprocess the dataset
    dataset = dataset.dropna()
    dataset["question"] = dataset["question"].apply(clean_text)
    dataset["answer"] = dataset["answer"].apply(clean_text)
    dataset = remove_repeating_sentences(dataset)

    #flask

    # Flask route for chatbot and dataset
    @app.route('/')
    def index():
    return render_template('index.html')

    @app.route('/chat', methods=['POST'])
    def chat():
    if request.method == 'POST':
        user_input = request.form['user_input']
        user_input = clean_text(user_input)

        # Check if the user input matches any question in the preprocessed dataset
        matching_row = dataset[dataset['question'] == user_input]
        
        if not matching_row.empty:
            # If a matching question is found, retrieve the corresponding answer
            bot_response = matching_row['answer'].values[0]
        else:
            # If no matching question is found, generate a response using the GPT-2 model
            input_ids = tokenizer.encode(user_input, return_tensors='pt')
            output = model.generate(input_ids, max_length=100, num_return_sequences=1)
            bot_response = tokenizer.decode(output[0], skip_special_tokens=True)
        
        return render_template('index.html', user_input=user_input, bot_response=bot_response)
    return render_template('index.html')

    @app.route('/dataset')
    def show_dataset():
    return render_template('dataset.html', data=dataset.to_dict(orient='records'))

    if __name__ == '__main__':
    app.run(debug=True)

