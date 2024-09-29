import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import openai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Download NLTK resources (if not already downloaded)
nltk.download('punkt')
nltk.download('stopwords')

class ChatbotPromptRefiner:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))

    def refine_prompt(self, prompt):
        """
        Refine a given chatbot prompt by removing stopwords and enhancing clarity.
        """
        # Tokenize the prompt
        words = word_tokenize(prompt.lower())
        
        # Remove stopwords
        filtered_words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        refined_prompt = ' '.join(filtered_words)
        return refined_prompt
    
    def use_openai_for_refinement(self, prompt):
        """
        Optionally enhance the prompt using OpenAI's language model.
        """
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=f"Improve this chatbot prompt: '{prompt}'",
            max_tokens=50
        )
        return response.choices[0].text.strip()
