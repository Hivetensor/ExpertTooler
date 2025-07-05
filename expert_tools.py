from langchain.tools import Tool

def create_expert_tools(smart_manager):
    """
    Creates a list of LangChain tools for each expert model.
    """
    expert_tools = []
    for expert_name, config in smart_manager.configs.items():
        tool = Tool(
            name=config["name"],
            description=config["description"],
            func=lambda q, expert_name=expert_name: smart_manager.get_expert(expert_name).generate([q]),
        )
        expert_tools.append(tool)
    return expert_tools
