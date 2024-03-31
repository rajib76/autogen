import os

from autogen import AssistantAgent, config_list_from_json, UserProxyAgent, GroupChat, GroupChatManager
from autogen.agentchat.contrib.capabilities.teachability import Teachability
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
os.environ['AUTOGEN_USE_DOCKER'] = "False"

filter_dict = {"model": ["gpt-4"]}  # GPT-3.5 is less reliable than GPT-4 at learning from user feedback.
config_list = config_list_from_json(env_or_file="OAI_CONFIG_LIST", filter_dict=filter_dict)
llm_config = {"config_list": config_list, "timeout": 120}

memory_expert = AssistantAgent(
    name="memory_expert",
    system_message="You will answer the user question *ONLY* if it can be answered based on a past similar "
                   "conversation, else return the most appropriate agent name from the available agents."
                   "<available_agents>"
                   "<agent>"
                   "<name>math_expert</name>"
                   "<description>answers math questions</description>"
                   "</agent>"
                   "<agent>"
                   "<name>physics_expert</name>"
                   "<description>answers physcis questions</description>"
                   "</agent>"
                   "</available_agents>",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

math_expert = AssistantAgent(
    name="math_expert",
    system_message="You are expert in answering maths question",
    llm_config=llm_config,
    human_input_mode="NEVER"
)

physics_expert = AssistantAgent(
    name="physics_expert",
    system_message="You are expert in answering physics question",
    llm_config=llm_config,
    human_input_mode="NEVER"
)


teachability = Teachability(
    verbosity=0,
    recall_threshold=0.5,
    reset_db=False,  # Use True to force-reset the memo DB, and False to use an existing DB.
    path_to_db_dir="./subject_db"  # Can be any path, but teachable agents in a group chat require unique paths.
)
teachability.add_to_agent(memory_expert)
# Instantiate a UserProxyAgent to represent the user. But in this notebook, all user input will be simulated.
user = UserProxyAgent(
    name="user",
    human_input_mode="NEVER",
    is_termination_msg=lambda x: True if "TERMINATE" in x.get("content") else False,
    max_consecutive_auto_reply=0,
    code_execution_config={
        "use_docker": False
    },
    # Please set use_docker=True if docker is available to run the generated code. Using docker is safer than running the generated code directly.
)


def state_transition(last_speaker, groupchat):
    messages = groupchat.messages
    if last_speaker is user:
        print("Speaker is user ")
        print("Message is ", messages)
        print("**********************")
        return memory_expert
    if last_speaker is memory_expert:
        if messages[-1]["content"].lower() not in ['math_expert','physics_expert']:
            print("Speaker is memory_expert none")
            print("Message is ", messages)
            print("**********************")
            return None
        else:
            print("Speaker is memory_expert ")
            print("Message is ", messages)
            print("**********************")
            if messages[-1]["content"].lower() == 'math_expert':
                return math_expert
            else:
                return physics_expert
        # if messages[-2]["name"] == 'math_expert':
        #     print("Speaker is memory_expert none")
        #     print("Message is ", messages)
        #     print("**********************")
        #     return None
        # else:
        #     print("Speaker is memory_expert ")
        #     print("Message is ", messages)
        #     print("**********************")
        #     return math_expert
    if last_speaker is math_expert:
        print("Speaker is math_expert ")
        print("Message is ", messages)
        print("**********************")
        return memory_expert

    else:
        print("Speaker is blank ")
        print("Message is ", messages)
        print("**********************")
        return None


# graph_dict = {}
# graph_dict[user] = [math_expert]
# graph_dict[math_expert] = [memory_expert]
# graph_dict[teachability.analyzer]=[user]

# text = "What is 2+2?"
# user.initiate_chat(math_expert, message=text, clear_history=True)

agents = [user,memory_expert,math_expert, physics_expert]

# create the groupchat
group_chat = GroupChat(agents=agents, messages=[],
                       max_round=10,
                       speaker_selection_method=state_transition)
# allowed_or_disallowed_speaker_transitions=graph_dict,
# allow_repeat_speaker=None, speaker_transitions_type="allowed")

# group_chat = GroupChat(agents=agents, messages=[], max_round=25, speaker_selection_method=state_transition)

# create the manager
manager = GroupChatManager(
    groupchat=group_chat,
    llm_config=llm_config,
    is_termination_msg=lambda x: x.get("content", "") and x.get("content", "").rstrip().endswith("TERMINATE"),
    code_execution_config=False,
)

# query = "What is 2+2?"
query = "What is refraction in physics?"
user.initiate_chat(manager, message=query)
