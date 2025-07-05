from langchain.agents import initialize_agent, AgentType
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

def create_orchestrator_agent(expert_tools, base_model_config):
    """
    Initializes and returns a LangChain ReAct agent.
    """
    print("Initializing base model for orchestrator...")
    
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=base_model_config.get("load_in_4bit", True),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_config["model_id"],
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_config["model_id"])

    base_model_pipeline = HuggingFacePipeline.from_model_and_tokenizer(
        model=model,
        tokenizer=tokenizer,
        task="text-generation",
        model_kwargs={"max_length": base_model_config.get("max_length", 2048)},
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    agent = initialize_agent(
        tools=expert_tools,
        llm=base_model_pipeline,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
    
    print("Orchestrator agent initialized.")
    return agent
