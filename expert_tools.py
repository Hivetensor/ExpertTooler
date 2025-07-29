class ExpertTool:
    """Simple tool class to replace langchain Tool"""
    def __init__(self, name, description, func):
        self.name = name
        self.description = description
        self.func = func

def create_expert_analysis_prompt(question, expert_type):
    """Create a specialized prompt for expert analysis"""
    base_prompt = f"""You are a {expert_type} providing concise analysis to help a general AI model answer a question. 

DO NOT provide the final answer directly. Instead, provide:
1. Key concepts/principles relevant to this question
2. Critical analysis or reasoning steps
3. Important considerations or gotchas
4. If the answer seems obvious to you, you may indicate your confidence

Keep your response under 100 words and focus on what would be most helpful for a general model to know.

Question: {question}

Expert Analysis:"""
    return base_prompt

def create_expert_tools(smart_manager):
    """
    Creates a list of tools for each expert model.
    
    Args:
        smart_manager: SmartModelManager instance containing expert configurations
        
    Returns:
        List of Tool objects
    """
    expert_tools = []
    
    def create_expert_func(expert_name, config):
        def expert_analysis(question):
            expert_prompt = create_expert_analysis_prompt(question, config["description"])
            return smart_manager.get_expert(expert_name)(expert_prompt)
        return expert_analysis
    
    for expert_name, config in smart_manager.configs.items():
        # Create a tool for each expert
        tool = ExpertTool(
            name=config["name"],
            description=config["description"],
            func=create_expert_func(expert_name, config)
        )
        expert_tools.append(tool)
        
    return expert_tools