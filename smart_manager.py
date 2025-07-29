import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

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

class SmartModelManager:
    def __init__(self, expert_configs, device="cuda"):
        self.experts = {}
        self.current_model = None
        self.configs = expert_configs
        self.tokenizer_cache = {}
        self.device = device

    def get_expert(self, expert_name):
        if self.current_model and self.current_model != expert_name:
            self.unload_all()

        if expert_name not in self.experts:
            print(f"Loading expert: {expert_name} on {self.device}...")
            config = self.configs[expert_name]

            if self.device == "dml":
                # VLLM doesn't support DirectML, fallback to CPU
                print("Warning: VLLM doesn't support DirectML, using CPU for experts")
                device = "cpu"
            else:
                device = self.device
                
            # Create VLLM model with conservative memory settings
            llm = LLM(
                model=config["model_id"],
                trust_remote_code=True,
                tensor_parallel_size=1,
                gpu_memory_utilization=0.7 if device != "cpu" else 0,  # More conservative
                enforce_eager=True,
                max_model_len=config.get("max_length", 2048),
                swap_space=2,  # Allow some CPU swap
                disable_log_stats=True  # Reduce overhead
            )
            
            # Create sampling parameters
            sampling_params = SamplingParams(
                temperature=0.1,
                top_p=0.9,
                max_tokens=config.get("max_length", 512),
                stop=["Question:", "Choices:", "\n\nQuestion"]
            )
            
            # Wrap in our custom wrapper
            self.experts[expert_name] = VLLMWrapper(llm, sampling_params)
            self.current_model = expert_name
            print(f"Expert {expert_name} loaded.")

        return self.experts[expert_name]

    def unload_all(self):
        print("Unloading all models...")
        # Properly cleanup VLLM models
        for expert_name, expert in self.experts.items():
            if hasattr(expert, 'llm') and hasattr(expert.llm, 'llm_engine'):
                try:
                    # Clean shutdown of VLLM engine
                    expert.llm.llm_engine._shutdown()
                except:
                    pass
        
        self.experts = {}
        self.current_model = None
        
        # Aggressive memory cleanup
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print("All models unloaded and cache cleared.")