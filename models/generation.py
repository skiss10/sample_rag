import openai

class Generator:
    def __init__(self, api_key, base_url=None, default_headers=None):
        """
        Initialize the Generator with the OpenAI API key and optional settings.

        :param api_key: Your OpenAI API key.
        :param base_url: The base URL for the OpenAI API (optional).
        :param default_headers: Default headers for the OpenAI API requests (optional).
        """
        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url
        if default_headers:
            openai.default_headers = default_headers

    def generate_response(self, context, prompt, model="gpt-3.5-turbo-16k"):
        """
        Generate a response using the OpenAI API with the provided context and prompt.

        :param context: List of context lines retrieved from the chat history.
        :param prompt: The user's prompt to generate a response for.
        :param model: The model to use for generation.
        :return: The generated response as a string.
        """
        # Combine context and prompt into a single input
        full_prompt = f"Context: {' '.join(context)}\nPrompt: {prompt}"
        
        print(full_prompt)

        # Generate a response using the OpenAI API
        completion = openai.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": full_prompt
                }
            ],
        )
        
        # Return the generated response
        return completion.choices[0].message.content.strip()
