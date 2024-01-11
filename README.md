# ESPOL Chatbot

## Overview

This repository contains the code for a chatbot designed to answer questions about the Escuela Superior Politecnica del Litoral (ESPOL). The chatbot leverages a retrieval-based QA system to provide accurate information drawn from various data sources related to the college.

## Features

- **Multilingual Support**: Answers questions in both English and Spanish.
- **Data Sources**: Utilizes CSV, PDF, and TXT files to draw comprehensive information.
- **Streamlit Interface**: Offers a web interface for easy interaction with users.
- **Customizable Responses**: Includes a mechanism to provide dynamic responses based on user input.

## How It Works

The chatbot is built using Streamlit and integrates the Langchain library to manage language models and information retrieval. When a user submits a question, the system uses vector similarity to find the most relevant information from the loaded data and provides an answer.

## Setup

1. Clone the repository.
2. Install the required dependencies listed in `requirements.txt`.
3. Set your OpenAI API key in the environment variables.
4. Run the Streamlit app using `streamlit run app.py`.

## Usage

Simply input your question in the chat interface and press 'Send'. The chatbot will process your question and return the most relevant answer it finds from its knowledge base.

## Contributing

Contributions to improve the chatbot are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.

