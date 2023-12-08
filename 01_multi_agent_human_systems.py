import os
from typing import Optional, List, Dict, Any, Union, Tuple

from autogen import UserProxyAgent, AssistantAgent, GroupChatManager, GroupChat, Agent
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')


# Creating the math assistant by inheriting the parent assistant agent class
class MathAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        message["content"] = message["name"] + ":" + message["content"]
        return super()._process_received_message(message, sender, silent)


# Creating the physics assistant by inheriting the parent assistant agent class
class PhysicsAssistantAgent(AssistantAgent):
    def _process_received_message(self, message, sender, silent):
        message["content"] = message["name"] + ":" + message["content"]
        return super()._process_received_message(message, sender, silent)


# Creating the physics assistant by inheriting the parent user agent class
class StudentProxyAgent(UserProxyAgent):
    def _process_received_message(self, message, sender, silent):
        message["content"] = message["name"] + ":" + message["content"]
        return super()._process_received_message(message, sender, silent)

    # Overriding this method from the ConversableAgent class to change the human prompt
    # when human input is required
    def get_human_input(self, prompt: str) -> str:
        """Get human input.

        Override this method to customize the way to get human input.

        Args:
            prompt (str): prompt for the human input.

        Returns:
            str: human input.
        """
        prompt = "Please type 'exit' to end the conversation or enter your next question\n\n"
        reply = input(prompt)
        return reply


config_list = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": OPENAI_API_KEY
    }
]

llm_config = {"config_list": config_list, "cache_seed": 42}

student_proxy = StudentProxyAgent(
    name="student",
    system_message="""A proxy for the student""",
    is_termination_msg=lambda x: x.get("content", " ").rstrip().endswith("TERMINATE"),
    human_input_mode="ALWAYS"
)

math_expert = MathAssistantAgent(
    name="math_expert",
    system_message="You are expert in maths and help answer all math questions. After answering ask for feedback "
                   "from user.",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

physics_expert = PhysicsAssistantAgent(
    name="physics_expert",
    system_message="""You are expert in physics and help answer all physics questions.
    After answering ask for feedback from user.""",
    llm_config=llm_config,
    human_input_mode="NEVER"
)
groupchat = GroupChat(agents=[student_proxy, math_expert, physics_expert], messages=[], max_round=12)
manager = GroupChatManager(name="teacher_assistant", groupchat=groupchat, llm_config=llm_config)

query = input("Ask your question:\n")
student_proxy.initiate_chat(manager, message=query)
