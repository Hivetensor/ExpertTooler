import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import time
from smart_manager import SmartModelManager
from expert_tools import create_expert_tools
from orchestrator import create_orchestrator_agent
from langchain_community.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
import torch

# --- Configuration ---
BASE_MODEL_CONFIG = {
    "model_id": "mistralai/Mistral-7B-Instruct-v0.1",
    "max_length": 2048,
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
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(row['choices'])])
    return f"{question}Choices:\n{choices}\nAnswer:"

def evaluate_mmlu_subset(model, dataset, num_questions=None, log_file=None):
    results = []
    
    # If num_questions is None or greater than the dataset length, evaluate on all questions
    eval_range = range(len(dataset)) if num_questions is None or num_questions > len(dataset) else range(num_questions)

    for i in tqdm(eval_range, desc="Evaluating"):
        row = dataset[i]
        prompt = format_question(row)
        
        try:
            if hasattr(model, 'run'): # It's an agent
                response = model.run(input=prompt)
            else: # It's a pipeline
                response = model(prompt)
            
            if log_file:
                with open(log_file, "a") as f:
                    f.write(f"--- Question {i} ---\n")
                    f.write(f"Prompt:\n{prompt}\n")
                    f.write(f"Response:\n{response}\n\n")

            # Extract only the generated text after the prompt
            generated_text = response.split("Answer:")[1] if "Answer:" in response else ""
            predicted_letter = generated_text.strip().upper()
            # Extract the first letter if the model provides a longer response
            if predicted_letter:
                predicted_letter = predicted_letter[0]

            predicted_answer = ord(predicted_letter) - ord('A') if predicted_letter else -1
            correct_answer = row['answer']
            
            results.append({
                "question": prompt,
                "predicted": predicted_letter,
                "correct": chr(65 + correct_answer),
                "is_correct": predicted_answer == correct_answer,
            })
        except Exception as e:
            print(f"Error processing question {i}: {e}")
            results.append({"question": prompt, "predicted": "ERROR", "correct": row['answer'], "is_correct": False})
            
    return pd.DataFrame(results)

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser(description="Run MMLU evaluation.")
    parser.add_argument("--mode", type=str, default="orchestration", choices=["baseline", "orchestration"],
                        help="Evaluation mode: 'baseline' or 'orchestration'.")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "dml"],
                        help="Device to run on: 'cuda' for NVIDIA GPUs, 'dml' for DirectML.")
    parser.add_argument("--num_questions", type=int, default=5,
                        help="Number of questions to evaluate per subject. If not provided, all questions will be evaluated.")
    args = parser.parse_args()

    print(f"Running evaluation in {args.mode} mode on {args.device} device.")

    # --- Create Log File ---
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"num_questions_{args.num_questions}_mode_{args.mode}_{timestamp}.txt")

    if args.mode == "baseline":
        # --- Initialize Baseline Model ---
        print("Initializing baseline model...")
        if args.device == "dml":
            from optimum.onnxruntime import ORTModelForCausalLM
            model = ORTModelForCausalLM.from_pretrained(BASE_MODEL_CONFIG["model_id"], provider="DmlExecutionProvider")
            tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_CONFIG["model_id"])
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=BASE_MODEL_CONFIG.get("max_length", 2048)
            )
            baseline_pipeline = HuggingFacePipeline(pipeline=pipe)
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
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_length=BASE_MODEL_CONFIG.get("max_length", 2048)
            )
            baseline_pipeline = HuggingFacePipeline(pipeline=pipe)
        model_to_evaluate = baseline_pipeline
        smart_manager = None
    else:
        # --- Initialize Managers and Tools for Orchestration ---
        smart_manager = SmartModelManager(EXPERT_CONFIGS, device=args.device)
        expert_tools = create_expert_tools(smart_manager)
        agent = create_orchestrator_agent(expert_tools, BASE_MODEL_CONFIG, device=args.device)
        model_to_evaluate = agent

    # --- Run Evaluation ---
    all_results = []
    domain_accuracies = {}

    for domain, subjects in MMLU_DOMAINS.items():
        print(f"\n--- Evaluating Domain: {domain} ---")
        domain_results = []
        for subject in subjects:
            print(f"  Loading MMLU subject: {subject}")
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                df_results = evaluate_mmlu_subset(model_to_evaluate, dataset, args.num_questions, log_file)
                domain_results.append(df_results)
                
                accuracy = df_results["is_correct"].mean()
                print(f"  Accuracy on {subject}: {accuracy:.2%}")
                
            except Exception as e:
                print(f"  Failed to load or evaluate {subject}: {e}")
        
        if domain_results:
            domain_df = pd.concat(domain_results)
            domain_accuracies[domain] = domain_df["is_correct"].mean()
            all_results.append(domain_df)

    # --- Display Results ---
    print("\n--- Evaluation Summary ---")
    summary_df = pd.DataFrame.from_dict(domain_accuracies, orient='index', columns=['Accuracy'])
    summary_df.index.name = 'Domain'
    print(summary_df.to_string(formatters={'Accuracy': '{:.2%}'.format}))

    if all_results:
        total_df = pd.concat(all_results)
        total_accuracy = total_df["is_correct"].mean()
        print(f"\nTotal Accuracy: {total_accuracy:.2%}")

    # --- Cleanup ---
    if smart_manager:
        smart_manager.unload_all()
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
