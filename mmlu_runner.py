import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from smart_manager import SmartModelManager
from expert_tools import create_expert_tools
from orchestrator import create_orchestrator_agent
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

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

def evaluate_mmlu_subset(model, dataset, num_questions=5):
    results = []
    for i in tqdm(range(min(num_questions, len(dataset))), desc="Evaluating"):
        row = dataset[i]
        prompt = format_question(row)
        
        try:
            if hasattr(model, 'run'): # It's an agent
                response = model.run(input=prompt)
            else: # It's a pipeline
                response = model(prompt)[0]['generated_text']

            predicted_answer = response.strip().upper()
            # Extract the first letter if the model provides a longer response
            if predicted_answer:
                predicted_answer = predicted_answer[0]

            correct_answer = row['answer']
            
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
    parser = argparse.ArgumentParser(description="Run MMLU evaluation.")
    parser.add_argument("--mode", type=str, default="orchestration", choices=["baseline", "orchestration"],
                        help="Evaluation mode: 'baseline' or 'orchestration'.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "dml"],
                        help="Device to run on: 'cuda' for NVIDIA GPUs, 'dml' for DirectML.")
    args = parser.parse_args()

    print(f"Running evaluation in {args.mode} mode on {args.device} device.")

    if args.mode == "baseline":
        # --- Initialize Baseline Model ---
        print("Initializing baseline model...")
        if args.device == "dml":
            from optimum.onnxruntime import ORTModelForCausalLM
            model = ORTModelForCausalLM.from_pretrained(BASE_MODEL_CONFIG["model_id"], provider="DmlExecutionProvider")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CONFIG["model_id"])
            baseline_pipeline = HuggingFacePipeline.from_model_and_tokenizer(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                model_kwargs={"max_length": BASE_MODEL_CONFIG.get("max_length", 2048)},
            )
        else: # cuda
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_CONFIG["model_id"],
                quantization_config=quantization_config,
                device_map="auto",
            )
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CONFIG["model_id"])
            baseline_pipeline = HuggingFacePipeline.from_model_and_tokenizer(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                model_kwargs={"max_length": BASE_MODEL_CONFIG.get("max_length", 2048)},
            )
        model_to_evaluate = baseline_pipeline
        smart_manager = None
    else:
        # --- Initialize Managers and Tools for Orchestration ---
        smart_manager = SmartModelManager(EXPERT_CONFIGS, device=args.device)
        expert_tools = create_expert_tools(smart_manager)
        agent = create_orchestrator_agent(expert_tools, BASE_MODEL_CONFIG, device=args.device)
        model_to_evaluate = agent

    # --- Run Evaluation ---
    for domain, subjects in MMLU_DOMAINS.items():
        print(f"\n--- Evaluating Domain: {domain} ---")
        for subject in subjects:
            print(f"  Loading MMLU subject: {subject}")
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                df_results = evaluate_mmlu_subset(model_to_evaluate, dataset)
                
                accuracy = df_results["is_correct"].mean()
                print(f"  Accuracy on {subject}: {accuracy:.2%}")
                print(df_results.head())
                
            except Exception as e:
                print(f"  Failed to load or evaluate {subject}: {e}")

    # --- Cleanup ---
    if smart_manager:
        smart_manager.unload_all()
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
