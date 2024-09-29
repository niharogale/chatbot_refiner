import os
import json
import io
import sys
from typing import List, Dict, Any
from openai import OpenAI  # Ensure you have installed the `openai` library
from groq import Groq  # Assuming you have installed the `groq` library

# Set your API keys here (ensure they are stored securely)
groq_api_key = os.getenv("GROQ_API_KEY") or "your_groq_api_key"
openrouter_api_key = os.getenv("OPENROUTER_API_KEY") or "your_openrouter_api_key"
openai_api_key = os.getenv("OPENAI_API_KEY") or "your_openai_api_key"

# Initialize clients
groq_client = Groq(api_key=groq_api_key)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=openrouter_api_key
)

openai_client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=openai_api_key
)

class Agent:

    def get_llm_response(client, prompt, openai_model="o1-preview", json_mode=False):

        if client == "openai":

            kwargs = {
                "model": openai_model,
                "messages": [{"role": "user", "content": prompt}]
            }

            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            response = openai_client.chat.completions.create(**kwargs)

        elif client == "groq":

            try:
                models = ["llama-3.1-8b-instant", "llama-3.1-70b-versatile", "llama3-70b-8192", "llama3-8b-8192", "gemma2-9b-it"]

                for model in models:

                    try:
                        kwargs = {
                            "model": model,
                            "messages": [{"role": "user", "content": prompt}]
                        }
                        if json_mode:
                            kwargs["response_format"] = {"type": "json_object"}

                        response = groq_client.chat.completions.create(**kwargs)

                        break

                    except Exception as e:
                        print(f"Error: {e}")
                        continue

            except Exception as e:
                print(f"Error: {e}")

                kwargs = {
                    "model": "meta-llama/llama-3.1-8b-instruct:free",
                    "messages": [{"role": "user", "content": prompt}]
                }

                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = openrouter_client.chat.completions.create(**kwargs)

        else:
            raise ValueError(f"Invalid client: {client}")

        return response.choices[0].message.content


    def evaluate_responses(self, prompt, reasoning_prompt=False, openai_model="o1-preview"):

        if reasoning_prompt:
            prompt = f"{prompt}\n\n{reasoning_prompt}."

        openai_response = self.get_llm_response("openai", prompt, openai_model)
        groq_response = self.get_llm_response("groq", prompt)

        print(f"OpenAI Response: {openai_response}")
        print(f"\n\nGroq Response: {groq_response}")

    def planner(self, user_query: str, client="groq", openaimodel="gpt-3.5-turbo") -> List[str]:
       prompt = f"""Given the user's query: '{user_query}', break down the query into as few subtasks as possible in order to answer the question.
       Each subtask is either a calculation or reasoning step. Never duplicate a task.
    
       Here are the only 2 actions that can be taken for each subtask:
           - generate_code: This action involves generating Python code and executing it in order to make a calculation or verification.
           - reasoning: This action involves providing reasoning for what to do to complete the subtask.
    
       Each subtask should begin with either "reasoning" or "generate_code".

       Keep in mind the overall goal of answering the user's query throughout the planning process.
    
       Return the result as a JSON list of strings, where each string is a subtask.
    
       Here is an example JSON response:
    
       {{
           "subtasks": ["Subtask 1", "Subtask 2", "Subtask 3"]
       }}
       """
       response = json.loads(self.get_llm_response(prompt=prompt, client=client, openai_model=openaimodel, json_mode=True))
       print(response)
       return response["subtasks"]

    def reasoner(self, user_query: str, subtasks: List[str], current_subtask: str, memory: List[Dict[str, Any]], client="groq", openaimodel="gpt-3.5-turbo") -> str:
       prompt = f"""Given the user's query (long-term goal): '{user_query}'
    
       Here are all the subtasks to complete in order to answer the user's query:
       <subtasks>
           {json.dumps(subtasks)}
       </subtasks>
    
       Here is the short-term memory (result of previous subtasks):
       <memory>
           {json.dumps(memory)}
       </memory>
    
       The current subtask to complete is:
       <current_subtask>
           {current_subtask}
       </current_subtask>
    
       - Provide concise reasoning on how to execute the current subtask, considering previous results.
       - Prioritize explicit details over assumed patterns
       - Avoid unnecessary complications in problem-solving
    
       Return the result as a JSON object with 'reasoning' as a key.
    
       Example JSON response:
       {{
           "reasoning": "2 sentences max on how to complete the current subtask."
       }}
       """
    
       response = json.loads(self.get_llm_response(prompt=prompt, client=client, openai_model=openaimodel, json_mode=True))
       return response["reasoning"]

    def actioner(self, user_query: str, subtasks: List[str], current_subtask: str, reasoning: str, memory: List[Dict[str, Any]], client="groq", openaimodel="gpt-3.5-turbo") -> Dict[str, Any]:
       prompt = f"""Given the user's query (long-term goal): '{user_query}'
    
       The subtasks are:
       <subtasks>
           {json.dumps(subtasks)}
       </subtasks>
    
       The current subtask is:
       <current_subtask>
           {current_subtask}
       </current_subtask>
    
       The reasoning for this subtask is:
       <reasoning>
           {reasoning}
       </reasoning>
    
       Here is the short-term memory (result of previous subtasks):
       <memory>
           {json.dumps(memory)}
       </memory>
    
       Determine the most appropriate action to take:
           - If the task requires a calculation or verification through code, use the 'generate_code' action.
           - If the task requires reasoning without code or calculations, use the 'reasoning' action.
    
       Consider the overall goal and previous results when determining the action.
    
       Return the result as a JSON object with 'action' and 'parameters' keys. The 'parameters' key should always be a dictionary with 'prompt' as a key.
    
       Example JSON responses:
    
       {{
           "action": "generate_code",
           "parameters": {{"prompt": "Write a function to calculate the area of a circle."}}
       }}
    
       {{
           "action": "reasoning",
           "parameters": {{"prompt": "Explain how to complete the subtask."}}
       }}
       """
    
       response = json.loads(self.get_llm_response(prompt=prompt, client=client, openai_model=openaimodel, json_mode=True))
       return response

    def evaluator(self, user_query: str, subtasks: List[str], current_subtask: str, action_info: Dict[str, Any], execution_result: Dict[str, Any], memory: List[Dict[str, Any]], client="qroq", openaimodel="gpt-3.5-turbo") -> Dict[str, Any]:
       prompt = f"""Given the user's query (long-term goal): '{user_query}'
    
       The subtasks to complete to answer the user's query are:
       <subtasks>
           {json.dumps(subtasks)}
       </subtasks>
    
       The current subtask to complete is:
       <current_subtask>
           {current_subtask}
       </current_subtask>
    
       The result of the current subtask is:
       <result>
           {action_info}
       </result>
    
       The execution result of the current subtask is:
       <execution_result>
           {execution_result}
       </execution_result>
    
       Here is the short-term memory (result of previous subtasks):
       <memory>
           {json.dumps(memory)}
       </memory>

       Evaluate if the result is a reasonable answer for the current subtask and makes sense in the context of the overall query.
    
       Return a JSON object with 'evaluation' (string) and 'retry' (boolean) keys.
    
       Example JSON response:
       {{
           "evaluation": "The result is a reasonable answer for the current subtask.",
           "retry": false
       }}
       """
    
       response = json.loads(self.get_llm_response(prompt=prompt, client=client, openai_model=openaimodel, json_mode=True))
       return response

    def final_answer_extractor(self, user_query: str, subtasks: List[str], memory: List[Dict[str, Any]], client="groq", openaimodel="gpt-3.5-turbo") -> str:
       prompt = f"""Given the user's query (long-term goal): '{user_query}'
    
       The subtasks completed to answer the user's query are:
       <subtasks>
           {json.dumps(subtasks)}
       </subtasks>
    
       The memory of the thought process (short-term memory) is:
       <memory>
           {json.dumps(memory)}
       </memory>
    
       Extract the final answer that directly addresses the user's query, from the memory.
       Provide only the essential information without unnecessary explanations.
    
       Return a JSON object with 'finalAnswer' as a key.
    
       Here is an example JSON response:
       {{
           "finalAnswer": "The final answer to the user's query, addressing all aspects of the question, based on the memory provided",
       }}
       """

       response = json.loads(self.get_llm_response(prompt=prompt, client=client, openai_model=openaimodel, json_mode=True))
       return response["finalAnswer"]

    def generate_and_execute_code(self, prompt: str, user_query: str, memory: List[Dict[str, Any]], client="groq", openaimodel="gpt-3.5-turbo") -> Dict[str, Any]:
       code_generation_prompt = f"""
    
       Generate Python code to implement the following task: '{prompt}'
    
       Here is the overall goal of answering the user's query: '{user_query}'

       Keep in mind the results of the previous subtasks, and use them to complete the current subtask.
       <memory>
           {json.dumps(memory)}
       </memory>

       Here are the guidelines for generating the code:
           - Return only the Python code, without any explanations or markdown formatting.
           - The code should always print or return a value
           - Don't include any backticks or code blocks in your response. Do not include ```python or ``` in your response, just give me the code.
           - Do not ever use the input() function in your code, use defined values instead.
           - Do not ever use NLP techniques in your code, such as importing nltk, spacy, or any other NLP library.
           - Don't ever define a function in your code, just generate the code to execute the subtask.
           - Don't ever provide the execution result in your response, just give me the code.
           - If your code needs to import any libraries, do it within the code itself.
           - The code should be self-contained and ready to execute on its own.
           - Prioritize explicit details over assumed patterns
           - Avoid unnecessary complications in problem-solving
       """
    
       generated_code = self.get_llm_response(prompt=code_generation_prompt, client=client, openai_model=openaimodel)
    
       print(f"\n\nGenerated Code: start|{generated_code}|END\n\n")
    
       old_stdout = sys.stdout
       sys.stdout = buffer = io.StringIO()
    
       exec(generated_code)
    
       sys.stdout = old_stdout
       output = buffer.getvalue()
    
       print(f"\n\n***** Execution Result: |start|{output.strip()}|end| *****\n\n")
    
       return {
           "generated_code": generated_code,
           "execution_result": output.strip()
       }

    def executor(self, action: str, parameters: Dict[str, Any], user_query: str, memory: List[Dict[str, Any]], client="groq", openaimodel="gpt-3.5-turbo") -> Any:
       if action == "generate_code":
           print(f"Generating code for: {parameters['prompt']}")
           return self.generate_and_execute_code(self=self, prompt=parameters["prompt"], user_query=user_query, memory=memory, client=client, openaimodel=openaimodel)
       elif action == "reasoning":
           return parameters["prompt"]
       else:
           return f"Action '{action}' not implemented"

    def autonomous_agent(self, user_query: str, client="groq", openaimodel="gpt-3.5-turbo") -> List[Dict[str, Any]]:
       memory = []
       subtasks = self.planner(self, user_query, client=client, openaimodel=openaimodel)
       for subtask in subtasks:
           reasoning = self.reasoner(self=self, user_query=user_query, subtasks=subtasks, current_subtask=subtask, memory=memory, client=client, openaimodel=openaimodel)
           action_info = self.actioner(self=self, user_query=user_query, subtasks=subtasks, current_subtask=subtask, reasoning=reasoning, memory=memory, client=client, openaimodel=openaimodel)
           execution_result =self.executor(self=self, action=action_info["action"], parameters=action_info["parameters"], user_query=user_query, memory=memory, client=client, openaimodel=openaimodel)
           evaluation = self.evaluator(self=self, user_query=user_query, subtasks=subtasks, current_subtask=subtask, action_info=action_info, execution_result=execution_result, memory=memory, client=client, openaimodel=openaimodel)
           if evaluation["retry"]:
               continue
           memory.append({
               "subtask": subtask,
               "result": execution_result,
               "evaluation": evaluation["evaluation"]
           })
       final_answer = self.final_answer_extractor(self=self, user_query=user_query, subtasks=subtasks, memory=memory, client=client, openaimodel=openaimodel)
       return final_answer
