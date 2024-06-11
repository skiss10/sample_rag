from models.retrieval import Retriever
from models.generation import Generator
import config

def main():
    retriever = Retriever()
    retriever.load_all_data()

    generator = Generator(api_key=config.OPENAI_API_KEY)

    prompt = "who is the contact for Tech Innovators?"

    context = retriever.retrieve_context(prompt)
    response = generator.generate_response(context, prompt)

    print("Prompt:", prompt)
    print("Response:", response)

if __name__ == "__main__":
    main()
