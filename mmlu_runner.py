import os
import pandas as pd
from datasets import load_dataset
from smart_manager import SmartModelManager
from expert_tools import create_expert_tools
from orchestrator import create_orchestrator_agent

# --- Configuration ---
BASE_MODEL_CONFIG = {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
    "max_length": 1024,
}

EXPERT_CONFIGS = {
    "math": {
        "name": "MathematicsExpert",
        "description": "Expert in mathematical reasoning, calculations, and symbolic math.",
        "model_id": "deepseek-ai/deepseek-math-7b-instruct",
    },
    "bio": {
        "name": "BiologyExpert",
        "description": "Expert in biology, medicine, and life sciences.",
        "model_id": "BioMistral/BioMistral-7B",
    },
    "chem": {
        "name": "ChemistryExpert",
        "description": "Expert in chemistry, molecular structures, and reactions.",
        "model_id": "epfl-llm/chemgpt-7b", # Fallback to BioMistral if needed
    },
    "code": {
        "name": "CodeExpert",
        "description": "Expert in programming, algorithms, and code generation.",
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
    },
}

MMLU_DOMAINS = {
    "STEM": ["abstract_algebra", "college_mathematics", "high_school_mathematics", "formal_logic"],
    "BioMed": ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "professional_medicine"],
    "Chemistry": ["college_chemistry", "high_school_chemistry"],
}

def format_question(row):
    question = f"Question: {row['question']}\n"
    choices = "\n".join([f"{chr(65+i)}. {row[f'choice{i}']}" for i in range(4)])
    return f"{question}Choices:\n{choices}\nAnswer:"

def evaluate_mmlu_subset(agent, dataset, num_questions=5):
    results = []
    for i in range(min(num_questions, len(dataset))):
        row = dataset[i]
        prompt = format_question(row)
        
        try:
            response = agent.run(input=prompt)
            predicted_answer = response.strip().upper()
            correct_answer = row['answer'].upper()
            
            results.append({
                "question": prompt,
                "predicted": predicted_answer,
                "correct": correct_answer,
                "is_correct": predicted_answer == correct_answer,
            })
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            results.append({"question": prompt, "predicted": "ERROR", "correct": row['answer'], "is_correct": False})
            
    return pd.DataFrame(results)

def main():
    # --- Initialize Managers and Tools ---
    smart_manager = SmartModelManager(EXPERT_CONFIGS)
    expert_tools = create_expert_tools(smart_manager)
    
    # --- Create Agent ---
    agent = create_orchestrator_agent(expert_tools, BASE_MODEL_CONFIG)

    # --- Run Evaluation ---
    for domain, subjects in MMLU_DOMAINS.items():
        print(f"\n--- Evaluating Domain: {domain} ---")
        for subject in subjects:
            print(f"  Loading MMLU subject: {subject}")
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                df_results = evaluate_mmlu_subset(agent, dataset)
                
                accuracy = df_results["is_correct"].mean()
                print(f"  Accuracy on {subject}: {accuracy:.2%}")
                print(df_results)
                
            except Exception as e:
                print(f"  Failed to load or evaluate {subject}: {e}")

    # --- Cleanup ---
    smart_manager.unload_all()
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
