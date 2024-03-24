import os

import autogen
from autogen import GroupChat, GroupChatManager
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

config_list = [
    {
        "model": "gpt-4-1106-preview",
        "api_key": OPENAI_API_KEY
    }
]

llm_config = {"config_list": config_list, "cache_seed": 42}

# create an AssistantAgent instance named "emotional_system"
emotional_system = autogen.AssistantAgent(
    name="emotional",
    system_message="You are the emotional system of the brain which monitors internal state to make a decision.You "
                   "always look for immediate gratification "
                   "Here is an example of a decision making process."
                   "situation: You are offered a delicious cake"
                   "perspective 1: Cake is full of sugar which is not healthy. It may cause harm to the body in future "
                   "perspective 2: The cake looks delicious, eating it will give immediate gratification"
                   "final decision: The cake looks delicious, eating it will give immediate gratification "
                   "*REMEMBER* you must always make a final decision.DO NOT provide any ambiguos decision",
    llm_config=llm_config,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
)

# create an AssistantAgent instance named "rational_system"
rational_system = autogen.AssistantAgent(
    name="rational",
    system_message="You are the rational system of the brain that cares about analysis of things in the outside world."
                   "You always think about long term consequences. "
                   "Here is an example of a decision making process."
                   "situation: You are offered a delicios cake"
                   "perspective 1: Cake is full of sugar which is not healthy. It may cause harm to the body in future "
                   "perspective 2: The cake looks delicious, eating it will give immediate gratification"
                   "final decision: Cake is full of sugar which is not healthy. It may cause harm to the body in future "
                   "*REMEMBER* you must always make a final decision.DO NOT provide any ambiguos decision",
    llm_config=llm_config,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
)

# create an AssistantAgent instance named "concious_mind"
concious_mind = autogen.AssistantAgent(
    name="concious",
    system_message="You are the judge who will evaluate the perspective of both rational and emotional systems to "
                   "take the final decision.You must make a final decision one way or the other. DO NOT provide any ambiguos decision.",
    llm_config=llm_config,
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
)

# create a UserProxyAgent instance named "user_proxy"
user_proxy = autogen.UserProxyAgent(
    name="user_proxy",
    system_message="You are a common man seeking the most appropriate answer to your question.",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=10,
    code_execution_config={
        "work_dir": "work_dir",
        "use_docker": False,
    },
)

# Creating the FSM transitions
graph_dict = {}
graph_dict[user_proxy] = [emotional_system]
graph_dict[emotional_system] = [rational_system]
graph_dict[rational_system] = [concious_mind]


agents = [user_proxy, emotional_system, rational_system, concious_mind]

# create the groupchat
group_chat = GroupChat(agents=agents, messages=[], max_round=25, allowed_or_disallowed_speaker_transitions=graph_dict, allow_repeat_speaker=None, speaker_transitions_type="allowed")

# create the manager
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

# query = "There is a train approaching its destination at a high speed. There are 5 people working on the train line. " \
#         "The train is going to hit them if we do not change the direction of the train. If I change the direction of " \
#         "the train, I am going to hit and kill one person. Should I divert the train to kill one person? "
# query = "I want to follow a healthy diet. I am offered to eat a delicious cheesecake. Should I eat it?"
query = "I need to take a loan to address a financial situation. Bank is ready to provide the loan but at a very high rate of interest. What should I do?"
# query = "The neighbour next door came to know of a secret that will cause my death, if I do not kill him first. What should I do?"
# query = "I can see both my wife and mother in danger. I can save only one. Who should I save?"
user_proxy.initiate_chat(manager, message=query)

