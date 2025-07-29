#!/usr/bin/env python3
"""
Context length analysis for MMLU runner models and prompts
"""

# Model context lengths (approximate)
models = {
    'Mistral 7B v0.3': 32768,  # 32k context
    'Llemma 7B': 2048,         # 2k context  
    'Meditron 7B': 2048,       # 2k context
    'BioMistral 7B': 4096,     # 4k context
    'CodeLlama 13B': 16384     # 16k context
}

print('Model Context Lengths:')
print('=' * 40)
for model, ctx_len in models.items():
    tokens_available = ctx_len - 512  # Reserve for generation
    print(f'{model:20} | {ctx_len:6} tokens | {tokens_available:6} usable')
    
print('\nPrompt Length Analysis:')
print('=' * 40)

# Estimate prompt lengths (rough token count)
routing_prompt = '''Analyze this question step by step to determine if expert analysis would be helpful:

Question: [MMLU Question - ~200 tokens]

Determine the primary domain and whether expert insight would be valuable:
- Mathematical calculations, proofs, equations, or complex math concepts? → MathematicsExpert
- Biological processes, anatomy, medicine, or life sciences? → BiologyExpert  
- Chemical reactions, molecular structures, or chemistry principles? → ChemistryExpert
- Programming concepts, algorithms, or computer science? → CodeExpert
- General knowledge, history, language, or simple facts? → NONE

Choose NONE if this is straightforward general knowledge that doesn't need specialized domain expertise.

Think step by step about which domain would provide the most valuable analysis.
Reasoning: [your reasoning here]
Expert: [expert name]'''

expert_analysis_prompt = '''You are a [Expert Type] providing concise analysis to help a general AI model answer a question. 

DO NOT provide the final answer directly. Instead, provide:
1. Key concepts/principles relevant to this question
2. Critical analysis or reasoning steps
3. Important considerations or gotchas
4. If the answer seems obvious to you, you may indicate your confidence

Keep your response under 100 words and focus on what would be most helpful for a general model to know.

Question: [MMLU Question - ~200 tokens]

Expert Analysis:'''

synthesis_prompt = '''You are a general AI model making the final decision based on expert analysis. The expert has provided context and insights, but YOU must make the final answer choice.

Use the expert's analysis to inform your reasoning, but think through the problem yourself and make your own decision.

For multiple-choice questions, you must respond with ONLY the following format:
FINAL ANSWER: [LETTER]

Question: [MMLU Question - ~200 tokens]

Expert Analysis from [Expert]: [Expert Response - ~100 tokens]

Now, using this expert context, analyze the question yourself and provide your final answer:
FINAL ANSWER:'''

# Rough token estimates (4 chars ≈ 1 token)
routing_tokens = len(routing_prompt) // 4 + 200
expert_tokens = len(expert_analysis_prompt) // 4 + 200  
synthesis_tokens = len(synthesis_prompt) // 4 + 200 + 100

print(f'Routing prompt:    ~{routing_tokens:3} tokens')
print(f'Expert analysis:   ~{expert_tokens:3} tokens') 
print(f'Synthesis prompt:  ~{synthesis_tokens:3} tokens')

print('\nContext Safety Analysis:')
print('=' * 40)
print('Model              | Context | Routing | Expert | Synthesis | Safe?')
print('-' * 65)
for model, ctx_len in models.items():
    usable = ctx_len - 512
    routing_safe = "OK" if routing_tokens < usable else "NO"
    expert_safe = "OK" if expert_tokens < usable else "NO" 
    synthesis_safe = "OK" if synthesis_tokens < usable else "NO"
    overall_safe = "OK" if all([routing_tokens < usable, expert_tokens < usable, synthesis_tokens < usable]) else "NO"
    
    print(f'{model:18} | {ctx_len:7} | {routing_safe:7} | {expert_safe:6} | {synthesis_safe:9} | {overall_safe:5}')

print('\nRecommendations:')
print('=' * 40)
print('- Llemma 7B and Meditron 7B have only 2048 tokens - may be tight')
print('- Consider shorter prompts for expert models')
print('- Mistral 7B v0.3 (orchestrator) has plenty of context (32k)')
print('- CodeLlama 13B should be fine with 16k context')