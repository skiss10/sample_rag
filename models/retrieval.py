from sentence_transformers import SentenceTransformer, util
import os
import csv
import PyPDF2

class Retriever:
    def __init__(self, model_name='paraphrase-MiniLM-L6-v2'):
        """
        Initialize the Retriever with a pre-trained SBERT model.

        :param model_name: The name of the pre-trained SBERT model to use for embeddings.
        """
        self.model = SentenceTransformer(model_name)
        self.chat_history = []
        self.vendor_data = []
        self.entertainment_data = []
        self.sponsor_data = []
        self.volunteer_data = []
        self.feedback_data = []
        self.permit_data = []
        self.status_data = []
        self.embeddings = None

    def load_text_data(self, file_path):
        """
        Load data from a text file and return as a list of lines.

        :param file_path: Path to the text file containing the data.
        :return: List of lines from the file.
        """
        with open(file_path, 'r') as f:
            return f.readlines()

    def load_pdf_data(self, file_path):
        """
        Load data from a PDF file and return as a list of lines.

        :param file_path: Path to the PDF file containing the data.
        :return: List of lines from the PDF.
        """
        content = ""
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page_num in range(len(reader.pages)):
                page = reader.pages[page_num]
                content += page.extract_text()
        return content.splitlines()

    def load_csv_data(self, file_path):
        """
        Load data from a CSV file and return as a list of formatted strings.

        :param file_path: Path to the CSV file containing the data.
        :return: List of formatted strings from the file.
        """
        formatted_data = []
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                formatted_row = "({})".format(
                    " ".join([f"{key.strip()}: {value}" for key, value in row.items()])
                )
                formatted_data.append(formatted_row)
        return formatted_data

    def load_data(self, file_path):
        """
        Load data from a file (text, PDF, or CSV) and return as a list of lines.

        :param file_path: Path to the file containing the data.
        :return: List of lines from the file.
        """
        file_extension = os.path.splitext(file_path)[1]
        if file_extension == '.txt':
            return self.load_text_data(file_path)
        elif file_extension == '.pdf':
            return self.load_pdf_data(file_path)
        elif file_extension == '.csv':
            return self.load_csv_data(file_path)
        else:
            raise ValueError("Unsupported file type")

    def load_all_data(self):
        """
        Load all data from respective files.
        """
        self.chat_history = self.load_data('data/chat_history.txt')
        self.vendor_data = self.load_data('data/vendors.txt')
        self.entertainment_data = self.load_data('data/entertainment.txt')
        self.sponsor_data = self.load_data('data/sponsors.txt')
        self.volunteer_data = self.load_data('data/volunteers.txt')
        self.feedback_data = self.load_data('data/feedback.txt')
        self.permit_data = self.load_data('data/permits.pdf')  # Example of loading a PDF file
        self.status_data = self.load_data('data/status.csv')  # Loading data from a CSV file
        
        # Combine all data sources into a single list for embeddings
        all_text_data = (
            self.chat_history + self.vendor_data + self.entertainment_data + 
            self.sponsor_data + self.volunteer_data + self.feedback_data + 
            self.permit_data
        )
        
        all_data = all_text_data + self.status_data
        
        # Create embeddings for all data sources
        self.embeddings = self.model.encode(all_data, convert_to_tensor=True)

    def retrieve_context(self, query, k=3, score_threshold=0.05):
        """
        Retrieve the most relevant lines from all data sources based on the query, including distance scores.

        :param query: The query to find relevant context for.
        :param k: The number of relevant lines to retrieve.
        :param score_threshold: The minimum score threshold to consider a context relevant.
        :return: List of the most relevant lines from all data sources with their scores.
        """
        # Create an embedding for the query using SBERT
        query_embedding = self.model.encode([query], convert_to_tensor=True)
        
        # Use semantic search to find the most relevant lines in all data sources
        hits = util.semantic_search(query_embedding, self.embeddings, top_k=k)[0]
        
        # Combine all data sources for reference
        all_text_data = (
            self.chat_history + self.vendor_data + self.entertainment_data + 
            self.sponsor_data + self.volunteer_data + self.feedback_data + 
            self.permit_data
        )
        
        all_data = all_text_data + self.status_data
        
        # Retrieve the corresponding lines from all data sources with their similarity scores
        relevant_context = [
            all_data[hit['corpus_id']].strip()
            for hit in hits if hit['score'] >= score_threshold
        ]
        
        return relevant_context

# Example usage
retriever = Retriever()
retriever.load_all_data()
query = "What are the permit requirements?"
context = retriever.retrieve_context(query)
print(context)
