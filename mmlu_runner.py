import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import time
import sys
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
        "model_id": "AI4Chem/ChemLLM-7B-Chat-1_5-SFT", # Fallback to BioMistral if needed
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

def evaluate_mmlu_subset(model, dataset, num_questions=None, log_file=None, mode="baseline"):
    results = []
    
    # If num_questions is None or greater than the dataset length, evaluate on all questions
    eval_range = range(len(dataset)) if num_questions is None or num_questions > len(dataset) else range(num_questions)

    # Redirect stdout to capture all debug output if in orchestration mode
    if mode == "orchestration" and log_file:
        class Tee:
            def __init__(self, *files):
                self.files = files
            def write(self, obj):
                for f in self.files:
                    f.write(obj)
                    f.flush()
            def flush(self):
                for f in self.files:
                    f.flush()
        
        log_handle = open(log_file, "a")
        original_stdout = sys.stdout
        sys.stdout = Tee(sys.stdout, log_handle)

    for i in tqdm(eval_range, desc="Evaluating"):
        row = dataset[i]
        prompt = format_question(row)
        
        if log_file:
            print(f"\n{'='*80}")
            print(f"QUESTION {i} - Correct Answer: {chr(65 + row['answer'])}")
            print(f"{'='*80}")
            print(f"Full Prompt:\n{prompt}")
            print(f"{'-'*80}")
        
        try:
            if hasattr(model, 'run'): # It's an orchestrator
                print("\n[ORCHESTRATOR PROCESSING]")
                response = model.run(input=prompt)
                
                # Get the routing decision from history
                if model.call_history:
                    last_call = model.call_history[-1]
                    print(f"\n[ROUTING SUMMARY]")
                    print(f"Expert Used: {last_call.get('expert_used', 'Unknown')}")
                    print(f"Routing Decision: {last_call.get('routing_decision', {})}")
                    
            else: # It's a baseline pipeline
                response = model(prompt)
                # Extract only the generated text after the prompt
                if "Answer:" in response:
                    response = response.split("Answer:")[-1]

            if log_file:
                print(f"\n[FINAL RESPONSE]: {response}")

            # Extract predicted letter
            predicted_letter = ""
            if "Answer:" in str(response):
                answer_part = str(response).split("Answer:")[-1].strip()
                if answer_part:
                    predicted_letter = answer_part[0].upper()
            else:
                # Try to find any letter A-D
                import re
                match = re.search(r'\b([A-D])\b', str(response))
                if match:
                    predicted_letter = match.group(1)

            predicted_answer = ord(predicted_letter) - ord('A') if predicted_letter in ['A', 'B', 'C', 'D'] else -1
            correct_answer = row['answer']
            
            is_correct = predicted_answer == correct_answer
            
            if log_file:
                print(f"\n[EVALUATION]")
                print(f"Predicted: {predicted_letter} (index: {predicted_answer})")
                print(f"Correct: {chr(65 + correct_answer)} (index: {correct_answer})")
                print(f"Result: {'CORRECT' if is_correct else 'WRONG'}")
            
            results.append({
                "question": prompt,
                "predicted": predicted_letter,
                "correct": chr(65 + correct_answer),
                "is_correct": is_correct,
            })
            
        except Exception as e:
            print(f"\n[ERROR] Processing question {i}: {e}")
            import traceback
            traceback.print_exc()
            results.append({"question": prompt, "predicted": "ERROR", "correct": chr(65 + row['answer']), "is_correct": False})
    
    # Restore stdout if we redirected it
    if mode == "orchestration" and log_file:
        sys.stdout = original_stdout
        log_handle.close()
            
    return pd.DataFrame(results)

def test_experts_directly(smart_manager, dataset, num_tests=3):
    """Test expert models directly without orchestration"""
    print("\n--- Direct Expert Testing ---")
    
    for expert_name, config in smart_manager.configs.items():
        print(f"\nTesting {config['name']}...")
        expert = smart_manager.get_expert(expert_name)
        
        # Find appropriate questions for this expert
        if expert_name == "math":
            subject = "college_mathematics"
        elif expert_name == "bio":
            subject = "college_biology"
        elif expert_name == "chem":
            subject = "college_chemistry"
        else:
            continue
            
        try:
            test_dataset = load_dataset("cais/mmlu", subject, split="test")
            for i in range(min(num_tests, len(test_dataset))):
                row = test_dataset[i]
                prompt = format_question(row)
                
                print(f"\nQuestion: {row['question'][:100]}...")
                print(f"Correct Answer: {chr(65 + row['answer'])}")
                
                response = expert(prompt)
                print(f"Expert Response: {response}")
                
        except Exception as e:
            print(f"Failed to test {expert_name}: {e}")

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser(description="Run MMLU evaluation.")
    parser.add_argument("--mode", type=str, default="orchestration", choices=["baseline", "orchestration", "test_experts"],
                        help="Evaluation mode: 'baseline', 'orchestration', or 'test_experts'.")
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
    log_file = os.path.join(log_dir, f"detailed_{args.mode}_n{args.num_questions}_{timestamp}.txt")

    # Initialize models
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
                trust_remote_code=True,
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
        
        if args.mode == "test_experts":
            # Just test experts directly
            test_dataset = load_dataset("cais/mmlu", "college_mathematics", split="test")
            test_experts_directly(smart_manager, test_dataset)
            smart_manager.unload_all()
            return
            
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
                df_results = evaluate_mmlu_subset(model_to_evaluate, dataset, args.num_questions, log_file, args.mode)
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

    # --- Display Orchestration Stats ---
    if hasattr(model_to_evaluate, 'get_routing_stats'):
        stats = model_to_evaluate.get_routing_stats()
        print("\n--- Swarm Intelligence Metrics ---")
        print(f"Expert utilization rate: {stats['expert_utilization_rate']:.2%}")
        print(f"Routing success rate: {stats['routing_success_rate']:.2%}")
        print("\nExpert usage distribution:")
        for expert, count in stats['expert_distribution'].items():
            print(f"  {expert}: {count}")
        
        # Save routing history
        history_file = os.path.join(log_dir, f"routing_history_{timestamp}.json")
        model_to_evaluate.save_call_history(history_file)
        print(f"\nRouting history saved to: {history_file}")

    # --- Cleanup ---
    if smart_manager:
        smart_manager.unload_all()
    print("\nEvaluation complete.")
    print(f"Detailed logs saved to: {log_file}")

if __name__ == "__main__":
    main()