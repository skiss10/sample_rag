# Sample RAG

## Overview

Sample RAG is a project that demonstrates the implementation of a retrieval-augmented generation (RAG) system. The project aims to showcase how to build an intelligent retrieval system that can pull relevant information from various data sources such as text files, PDFs, and CSV files and use this information to generate meaningful responses.

## Features

- Load and process data from text files, PDFs, and CSV files.
- Generate semantic embeddings using a pre-trained SBERT model.
- Perform semantic search to retrieve relevant information based on a query.
- Support for various data types to enhance the versatility of the retrieval system.

## File Structure

- `retrieval.py`: Contains the `Retriever` class that handles loading data, creating embeddings, and retrieving relevant information.
- `generator.py`: Contains the `Generator` class that handles generating responses based on the retrieved context.
- `main.py`: Contains the main script that uses both the `Retriever` and `Generator` classes to process input and generate output.
- `data/`: Directory containing sample data files (text, PDF, and CSV) used by the `Retriever` class.
- `requirements.txt`: List of dependencies required to run the project.

## Data Loading Methods

The `Retriever` class supports loading data from various sources:

- **Text Files**: Reads data line-by-line from `.txt` files.
- **PDF Files**: Extracts text from `.pdf` files.
- **CSV Files**: Reads and formats data from `.csv` files.

## Example Data

The `data/` directory contains example files for testing the retrieval system:

- `chat_history.txt`
- `vendors.txt`
- `entertainment.txt`
- `sponsors.txt`
- `volunteers.txt`
- `feedback.txt`
- `permits.pdf`
- `status.csv`

