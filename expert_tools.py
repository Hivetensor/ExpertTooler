class ExpertTool:
    """Simple tool class to replace langchain Tool"""
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

def create_expert_tools(smart_manager):
    """
    Creates a list of tools for each expert model.
    
    Args:
        smart_manager: SmartModelManager instance containing expert configurations
        
    Returns:
        List of Tool objects
    """
    expert_tools = []
    
    for expert_name, config in smart_manager.configs.items():
        # Create a tool for each expert
        tool = ExpertTool(
            name=config["name"],
            description=config["description"],
            func=lambda q, expert_name=expert_name: smart_manager.get_expert(expert_name)(q),
        )
        expert_tools.append(tool)
        
    return expert_tools