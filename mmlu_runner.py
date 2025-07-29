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
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
import json 

# --- Configuration ---
BASE_MODEL_CONFIG = {
    "model_id": "mistralai/mistral-7b-instruct-v0.3",  # ~35GB in 4-bit
    "max_length": 2048,
}

EXPERT_CONFIGS = {
    "math": {
        "name": "MathematicsExpert",
        "description": "Expert in mathematical reasoning, calculations, and symbolic math.",
        "model_id": "EleutherAI/llemma_7b",  # RL-tuned version
    },
    "bio": {
        "name": "BiologyExpert",
        "description": "Expert in biology, medicine, and life sciences.",
        "model_id": "epfl-llm/meditron-7b",  # Better medical knowledge
    },
    "chem": {
        "name": "ChemistryExpert",
        "description": "Expert in chemistry, molecular structures, and reactions.",
        "model_id": "BioMistral/BioMistral-7B",
    },
    "code": {
        "name": "CodeExpert",
        "description": "Expert in programming, algorithms, and code generation.",
        "model_id": "codellama/CodeLlama-13b-Instruct-hf",  # ~8GB in 4-bit
    },
}


MMLU_DOMAINS = {
    "STEM": ["abstract_algebra", "college_mathematics", "high_school_mathematics", "formal_logic"],
    "BioMed": ["anatomy", "clinical_knowledge", "college_biology", "college_medicine", "professional_medicine"],
    "Chemistry": ["college_chemistry", "high_school_chemistry"],
}

class VLLMWrapper:
    """Wrapper to make VLLM compatible with the existing interface"""
    
    def __init__(self, llm, sampling_params):
        self.llm = llm
        self.sampling_params = sampling_params
    
    def generate(self, prompt):
        """Generate response using VLLM"""
        outputs = self.llm.generate([prompt], self.sampling_params)
        if outputs and len(outputs) > 0:
            return outputs[0].outputs[0].text.strip()
        return ""
    
    def __call__(self, prompt):
        """Make the wrapper callable like the old pipeline"""
        return self.generate(prompt)

def benchmark_individual_models(smart_manager, base_model_config, device="cuda"):
    """Test each model individually with proper subject-level comparisons"""
    results = {}
    
    print("\n" + "="*80)
    print("INDIVIDUAL MODEL BENCHMARKING - SUBJECT BY SUBJECT")
    print("="*80)
    
    # Define expert-to-subject mappings for fair comparison
    expert_subject_mapping = {
        "math": ["abstract_algebra", "college_mathematics", "high_school_mathematics"],
        "bio": ["anatomy", "college_biology", "college_medicine"], 
        "chem": ["college_chemistry", "high_school_chemistry"],
    }
    
    # Test baseline model on each subject individually
    print("\n1. Testing baseline model on individual subjects...")
    baseline_pipeline = create_baseline_model(base_model_config, device)
    
    # Test baseline on each expert's subjects for direct comparison
    for expert_name, subjects in expert_subject_mapping.items():
        print(f"\n  Baseline performance on {expert_name} subjects:")
        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                df = evaluate_mmlu_subset(baseline_pipeline, dataset, num_questions=10)
                accuracy = df["is_correct"].mean()
                results[f"baseline_{subject}"] = accuracy
                print(f"    {subject}: {accuracy:.2%}")
            except Exception as e:
                print(f"    {subject}: ERROR - {e}")
    
    # Clean up baseline model
    print("\n  Unloading baseline model...")
    import gc
    del baseline_pipeline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Test each expert on their specialized subjects
    print("\n2. Testing experts on their specialized subjects...")
    for expert_name, subjects in expert_subject_mapping.items():
        if expert_name not in smart_manager.configs:
            continue
            
        print(f"\n  {expert_name.upper()} Expert performance:")
        
        # Load expert (this will unload previous experts automatically)
        expert = smart_manager.get_expert(expert_name)
        
        for subject in subjects:
            try:
                dataset = load_dataset("cais/mmlu", subject, split="test")
                df = evaluate_mmlu_subset(expert, dataset, num_questions=10)
                accuracy = df["is_correct"].mean()
                results[f"{expert_name}_expert_{subject}"] = accuracy
                print(f"    {subject}: {accuracy:.2%}")
                
                # Calculate improvement over baseline
                baseline_key = f"baseline_{subject}"
                if baseline_key in results:
                    improvement = accuracy - results[baseline_key]
                    print(f"      vs baseline: {improvement:+.2%}")
                    
            except Exception as e:
                print(f"    {subject}: ERROR - {e}")
        
        # Explicitly unload this expert before moving to next
        print(f"  Unloading {expert_name} expert...")
        smart_manager.unload_all()
    
    # Summary comparison
    print("\n3. SUMMARY - Expert vs Baseline by Subject:")
    print("-" * 60)
    print(f"{'Subject':<25} | {'Baseline':<10} | {'Expert':<10} | {'Improvement':<12}")
    print("-" * 60)
    
    for expert_name, subjects in expert_subject_mapping.items():
        if expert_name not in smart_manager.configs:
            continue
            
        for subject in subjects:
            baseline_key = f"baseline_{subject}"
            expert_key = f"{expert_name}_expert_{subject}"
            
            if baseline_key in results and expert_key in results:
                baseline_acc = results[baseline_key]
                expert_acc = results[expert_key]
                improvement = expert_acc - baseline_acc
                
                print(f"{subject:<25} | {baseline_acc:<10.2%} | {expert_acc:<10.2%} | {improvement:+<12.2%}")
                
                # Store summary results
                results[f"improvement_{subject}"] = improvement
    
    return results

def create_baseline_model(base_model_config, device="cuda"):
    """Helper to create baseline model using VLLM"""
    if device == "dml":
        # VLLM doesn't support DirectML, fallback to CPU
        print("Warning: VLLM doesn't support DirectML, using CPU")
        device = "cpu"
    
    # Create VLLM model with conservative memory settings
    llm = LLM(
        model=base_model_config["model_id"],
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.7 if device != "cpu" else 0,  # More conservative
        enforce_eager=True,
        swap_space=2,  # Allow some CPU swap
        disable_log_stats=True  # Reduce overhead
    )
    
    # Create sampling parameters
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.9,
        max_tokens=base_model_config.get("max_length", 512),
        stop=["Question:", "Choices:", "\n\nQuestion"]
    )
    
    return VLLMWrapper(llm, sampling_params)


def format_question(row):
    question = f"Question: {row['question']}\n"
    choices = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(row['choices'])])
    return f"{question}Choices:\n{choices}\nAnswer:"

def evaluate_mmlu_subset(model, dataset, num_questions=None, log_file=None, mode="baseline"):
    results = []
    
    # If num_questions is None or greater than the dataset length, evaluate on all questions
    eval_range = range(len(dataset)) if num_questions is None or num_questions > len(dataset) else range(num_questions)

    # Redirect stdout to capture all debug output if in orchestration mode
    

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
                    
            else: # It's a baseline VLLM model
                response = model.generate(prompt)
                # VLLM returns the generated text without the prompt
                if isinstance(response, list) and len(response) > 0:
                    response = response[0]

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
                
                if hasattr(expert, 'generate'):
                    response = expert.generate(prompt)
                else:
                    response = expert(prompt)
                print(f"Expert Response: {response}")
                
        except Exception as e:
            print(f"Failed to test {expert_name}: {e}")

def main():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    parser = argparse.ArgumentParser(description="Run MMLU evaluation.")
    parser.add_argument("--mode", type=str, default="orchestration", 
                        choices=["baseline", "orchestration", "test_experts", "benchmark_all"],  # Added benchmark_all
                        help="Evaluation mode")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "dml"],
                        help="Device to run on: 'cuda' for NVIDIA GPUs, 'dml' for DirectML.")
    parser.add_argument("--num_questions", type=int, default=5,
                        help="Number of questions to evaluate per subject. If not provided, all questions will be evaluated.")
    args = parser.parse_args()

    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"detailed_{args.mode}_n{args.num_questions}_{timestamp}.txt")


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

    print(f"Running evaluation in {args.mode} mode on {args.device} device.")

    # --- Create Log File ---

    # Initialize models
    if args.mode == "benchmark_all":
        # Run individual benchmarks first
        smart_manager = SmartModelManager(EXPERT_CONFIGS, device=args.device)
        
        benchmark_results = benchmark_individual_models(
            smart_manager, 
            BASE_MODEL_CONFIG, 
            device=args.device
        )
        
        # Save results
        results_file = os.path.join(log_dir, f"benchmark_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        
        print(f"\nResults saved to: {results_file}")
        
        # Analyze results by subject for better insights
        print("\n" + "="*80)
        print("EXPERT EFFECTIVENESS ANALYSIS")
        print("="*80)
        
        expert_subjects = {
            "math": ["abstract_algebra", "college_mathematics", "high_school_mathematics"],
            "bio": ["anatomy", "college_biology", "college_medicine"], 
            "chem": ["college_chemistry", "high_school_chemistry"],
        }
        
        overall_improvements = {}
        for expert_name, subjects in expert_subjects.items():
            improvements = []
            print(f"\n{expert_name.upper()} Expert Analysis:")
            
            for subject in subjects:
                improvement_key = f"improvement_{subject}"
                if improvement_key in benchmark_results:
                    improvement = benchmark_results[improvement_key]
                    improvements.append(improvement)
                    status = "✓" if improvement > 0 else "✗" if improvement < -0.01 else "~"
                    print(f"  {subject:<25}: {improvement:+.2%} {status}")
            
            if improvements:
                avg_improvement = sum(improvements) / len(improvements)
                overall_improvements[expert_name] = avg_improvement
                print(f"  Average improvement: {avg_improvement:+.2%}")
                
                if avg_improvement < -0.01:
                    print(f"  ⚠️  WARNING: {expert_name} expert is WORSE than baseline!")
                elif avg_improvement > 0.01:
                    print(f"  ✓ {expert_name} expert shows positive improvement")
                else:
                    print(f"  ~ {expert_name} expert shows marginal difference")
        
        # Overall recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION:")
        effective_experts = [name for name, improvement in overall_improvements.items() if improvement > 0.01]
        if effective_experts:
            print(f"✓ Use orchestration with experts: {', '.join(effective_experts)}")
        else:
            print("✗ Experts do not show clear benefit - consider using baseline only")
        
        smart_manager.unload_all()
        return
    

    elif args.mode == "baseline":
        # --- Initialize Baseline Model ---
        print("Initializing baseline model with VLLM...")
        model_to_evaluate = create_baseline_model(BASE_MODEL_CONFIG, device=args.device)
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