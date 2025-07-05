from langchain.tools import Tool

def create_expert_tools(smart_manager):
    """
    Creates a list of LangChain tools for each expert model.
    
    Args:
        smart_manager: SmartModelManager instance containing expert configurations
        
    Returns:
        List of LangChain Tool objects
    """
    expert_tools = []
    
    for expert_name, config in smart_manager.configs.items():
        # Create a tool for each expert
        tool = Tool(
            name=config["name"],
            description=config["description"],
            # Fix: HuggingFacePipeline objects are callable, not .generate()
            func=lambda q, expert_name=expert_name: smart_manager.get_expert(expert_name)(q),
        )
        expert_tools.append(tool)
        
    return expert_tools