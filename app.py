from flask import Flask, render_template, request, jsonify
from agent import Agent

# Create a Flask app
app = Flask(__name__)

class ModelManager:
    def __init__(self):
        # Initialize available models and agents
        self.clients = {
            'client_1': 'groq',
            'client_2': 'openai'
        }
        self.models = {
            'model_1': 'gpt-3.5-turbo',
            'model_2': 'gpt-4-turbo',
        }
        self.agents = {
            'agent_1': 'With Agent',
            'agent_2': 'Without Agent'
        }

    def generate_response(self, client_choice, model_choice, prompt, use_agent=False):
        # Implement model and agent logic
        if client_choice == 'groq':
            if use_agent:
                response = Agent.autonomous_agent(self=Agent, user_query=prompt)
            else:
                response = Agent.get_llm_response(client=client_choice, prompt=prompt)
        else:
            if use_agent:
                response = Agent.autonomous_agent(self=Agent, user_query=prompt, client=client_choice, openai_model=model_choice)
            else:
                response = Agent.get_llm_response(client=client_choice, prompt=prompt, openai_model=model_choice)
        return response

# Instantiate ModelManager
model_manager = ModelManager()

@app.route('/')
def index():
    # Render the index.html template with model and agent options
    return render_template('index.html', models=model_manager.models, agents=model_manager.agents, clients=model_manager.clients)

@app.route('/generate', methods=['POST'])
def generate():
    # Get data from the form submission
    client_choice = request.form.get('client_choice')  # Extract the client choice
    model_choice = request.form.get('model_choice')
    prompt = request.form.get('prompt')
    use_agent = request.form.get('use_agent') == 'yes'

    # Generate response using the chosen model/agent
    response = model_manager.generate_response(model_manager.clients[client_choice], model_choice, prompt, use_agent)

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
