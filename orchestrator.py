from langchain.schema import BaseOutputParser
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch
import re
import json

class RoutingDecisionParser(BaseOutputParser):
    """Parser for routing decisions with fallback handling"""
    def parse(self, text: str):
        decision = {
            'use_expert': False,
            'expert': 'NONE',
            'reasoning': ''
        }
        
        try:
            # Look for "Expert:" in the output
            if "Expert:" in text:
                # Extract everything after "Expert:" until newline or "Answer:"
                expert_line = text.split("Expert:")[-1]
                # Stop at "Answer:" if present
                if "Answer:" in expert_line:
                    expert_line = expert_line.split("Answer:")[0]
                expert_name = expert_line.strip().split('\n')[0].strip()
                
                # Check if it's a valid expert name
                valid_experts = ['MathematicsExpert', 'BiologyExpert', 'ChemistryExpert', 'CodeExpert', 'NONE']
                if expert_name in valid_experts:
                    if expert_name == 'NONE':
                        decision['use_expert'] = False
                        decision['expert'] = 'NONE'
                    else:
                        decision['use_expert'] = True
                        decision['expert'] = expert_name
                    decision['reasoning'] = 'Expert identified from routing'
                    
            # Fallback: look for expert names anywhere in the text
            if not decision['use_expert']:
                text_lower = text.lower()
                if 'mathematicsexpert' in text_lower:
                    decision = {'use_expert': True, 'expert': 'MathematicsExpert', 'reasoning': 'Math expert detected'}
                elif 'biologyexpert' in text_lower:
                    decision = {'use_expert': True, 'expert': 'BiologyExpert', 'reasoning': 'Biology expert detected'}
                elif 'chemistryexpert' in text_lower:
                    decision = {'use_expert': True, 'expert': 'ChemistryExpert', 'reasoning': 'Chemistry expert detected'}
                elif 'codeexpert' in text_lower:
                    decision = {'use_expert': True, 'expert': 'CodeExpert', 'reasoning': 'Code expert detected'}
                    
        except Exception as e:
            print(f"[PARSER ERROR]: {e}")
            
        return decision

class SwarmOrchestrator:
    """Orchestrator that intelligently routes questions to expert models or answers directly"""
    
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        
        # Routing decision prompt - simplified for better parsing
        self.routing_prompt = PromptTemplate(
            template="""Analyze this question step by step:

Question: {question}

First, identify the primary domain:
- Is this primarily about mathematical calculations, proofs, or formulas? → MathematicsExpert
- Is this about biological systems, medicine, or life sciences? → BiologyExpert  
- Is this about chemical reactions, molecules, or chemistry concepts? → ChemistryExpert
- Is this about programming, algorithms, or code? → CodeExpert
- Is this general knowledge that doesn't require specialized expertise? → NONE

Think step by step about which domain this belongs to.
Reasoning: [your reasoning here]
Expert: [expert name]"""
    )

        
        # Direct answer prompt (when no expert is used)
        self.direct_prompt = PromptTemplate(
            input_variables=["question"],
            template="""Answer the following question directly. If it's a multiple choice question, analyze all options and provide your answer in the format:
FINAL ANSWER: [A/B/C/D]

Question: {question}

Answer:"""
        )
        
        # Synthesis prompt (when expert is used)
        self.synthesis_prompt = PromptTemplate(
            input_variables=["question", "expert_response", "expert_name"],
            template="""You are an expert synthesizer. Your job is to take a question and an expert's analysis and output a final answer.
For multiple-choice questions, you must respond with ONLY the following format:
FINAL ANSWER: [LETTER]

Question: {question}
{expert_name}'s Analysis: {expert_response}

Based on the analysis, what is the final answer?
Final Answer:"""
        )
        
        # Initialize chains
        self.router = LLMChain(
            llm=self.llm, 
            prompt=self.routing_prompt,
            output_parser=RoutingDecisionParser()
        )
        self.direct_chain = LLMChain(llm=self.llm, prompt=self.direct_prompt)
        self.synthesis_chain = LLMChain(llm=self.llm, prompt=self.synthesis_prompt)
        
        # Research metrics tracking
        self.call_history = []
        self.debug_mode = True  # Set to False to reduce verbosity
        
    def run(self, input, **kwargs):
        """Main orchestration logic"""
        
        # Step 1: Make routing decision
        try:
            routing_text = self.router.llm(self.routing_prompt.format(question=input))
            decision = self.router.output_parser.parse(routing_text)
            
            if self.debug_mode:
                print(f"\n[ROUTING DECISION]")
                print(f"Use Expert: {decision['use_expert']}")
                print(f"Expert: {decision['expert']}")
                print(f"Reasoning: {decision['reasoning']}")
                
        except Exception as e:
            print(f"[ERROR] Routing failed: {e}")
            decision = {'use_expert': False, 'expert': 'NONE', 'reasoning': 'Routing error - using direct answer'}
        
        # Record routing decision
        call_record = {
            'question': input,
            'routing_decision': decision,
            'expert_used': 'NONE',
            'final_response': ''
        }
        
        # Step 2: Execute based on routing decision
        if decision['use_expert'] and decision['expert'] in self.tools:
            # Use expert model
            expert_name = decision['expert']
            call_record['expert_used'] = expert_name
            
            if self.debug_mode:
                print(f"\n[CALLING EXPERT: {expert_name}]")
            
            try:
                # Call the expert tool
                expert_response = self.tools[expert_name].func(input)
                
                if self.debug_mode:
                    print(f"[EXPERT RESPONSE]: {expert_response}")
                
                # Synthesize final answer using expert response
                final_response = self.synthesis_chain.run(
                    question=input,
                    expert_response=expert_response,
                    expert_name=expert_name
                )
                
            except Exception as e:
                print(f"[ERROR] Expert call failed: {e}")
                # Fallback to direct answer
                final_response = self.direct_chain.run(question=input)
                call_record['expert_used'] = 'NONE (expert failed)'
                
        else:
            # Answer directly without expert
            if self.debug_mode:
                print(f"\n[DIRECT ANSWER - No expert needed]")
                
            final_response = self.direct_chain.run(question=input)
            call_record['expert_used'] = 'NONE'
        
        # Step 3: Clean up response for multiple choice
        if "Choices:" in input and any(letter in input for letter in ["A.", "B.", "C.", "D."]):
            # Extract answer letter from response
            answer_match = re.search(r'FINAL ANSWER:\s*([A-D])', final_response, re.IGNORECASE)
            if answer_match:
                final_response = f"Answer: {answer_match.group(1)}"
            else:
                # If formatting is incorrect, raise an error.
                raise ValueError("Response format from synthesis chain is invalid.")
                    
        call_record['final_response'] = final_response
        self.call_history.append(call_record)
        
        if self.debug_mode:
            print(f"[FINAL RESPONSE]: {final_response}")
            
        return final_response
    
    def invoke(self, input_dict, **kwargs):
        """Support for newer LangChain versions"""
        if isinstance(input_dict, dict) and 'input' in input_dict:
            return self.run(input_dict['input'], **kwargs)
        return self.run(input_dict, **kwargs)
    
    def get_routing_stats(self):
        """Get routing statistics for research analysis"""
        total = len(self.call_history)
        if total == 0:
            return {
                'total_questions': 0,
                'expert_utilization_rate': 0,
                'expert_distribution': {},
                'routing_success_rate': 0
            }
            
        expert_calls = sum(1 for h in self.call_history if h.get('expert_used') not in ['NONE', 'NONE (expert failed)'])
        failed_calls = sum(1 for h in self.call_history if 'failed' in h.get('expert_used', ''))
        
        # Count by expert
        expert_counts = {}
        for expert in ['MathematicsExpert', 'BiologyExpert', 'ChemistryExpert', 'CodeExpert', 'NONE']:
            expert_counts[expert] = sum(1 for h in self.call_history if h.get('expert_used') == expert)
            
        return {
            'total_questions': total,
            'expert_utilization_rate': expert_calls / total,
            'expert_distribution': expert_counts,
            'routing_success_rate': (total - failed_calls) / total,
            'failed_expert_calls': failed_calls
        }
    
    def save_call_history(self, filepath):
        """Save detailed call history for analysis"""
        with open(filepath, 'w') as f:
            json.dump(self.call_history, f, indent=2)

def create_orchestrator_agent(expert_tools, base_model_config, device="cuda"):
    """
    Creates a swarm orchestrator that intelligently routes questions to expert models.
    
    Args:
        expert_tools: List of LangChain tools representing expert models
        base_model_config: Configuration for the base orchestrator model
        device: Device to run on ("cuda" or "dml")
    
    Returns:
        SwarmOrchestrator instance
    """
    print(f"Initializing swarm orchestrator on {device}...")
    
    # Initialize base model for orchestration
    if device == "dml":
        from optimum.onnxruntime import ORTModelForCausalLM
        model = ORTModelForCausalLM.from_pretrained(
            base_model_config["model_id"], 
            provider="DmlExecutionProvider"
        )
    else:  # cuda
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=base_model_config.get("load_in_4bit", True),
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model_config["model_id"],
            quantization_config=quantization_config,
            trust_remote_code=True,
            device_map="auto",
        )
    
    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_config["model_id"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=base_model_config.get("max_length", 512),
        temperature=0.1,  # Low temperature for more deterministic routing
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    # Wrap in LangChain LLM
    base_llm = HuggingFacePipeline(pipeline=pipe)
    
    # Create and return orchestrator
    orchestrator = SwarmOrchestrator(base_llm, expert_tools)
    print("Swarm orchestrator initialized successfully.")
    
    return orchestrator
