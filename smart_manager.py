import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from langchain_community.llms import HuggingFacePipeline

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
                from optimum.onnxruntime import ORTModelForCausalLM
                model = ORTModelForCausalLM.from_pretrained(config["model_id"], provider="DmlExecutionProvider")
            else: # cuda
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=config.get("load_in_4bit", True),
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                )
                model = AutoModelForCausalLM.from_pretrained(
                    config["model_id"],
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    device_map="auto",
                )
            
            if config["model_id"] not in self.tokenizer_cache:
                self.tokenizer_cache[config["model_id"]] = AutoTokenizer.from_pretrained(config["model_id"])
            
            tokenizer = self.tokenizer_cache[config["model_id"]]
            
            # Ensure tokenizer has pad_token
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            # Create the pipeline first
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=config.get("max_length", 512),
                temperature=0.1,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
            
            # Then wrap it in HuggingFacePipeline
            self.experts[expert_name] = HuggingFacePipeline(pipeline=pipe)
            self.current_model = expert_name
            print(f"Expert {expert_name} loaded.")

        return self.experts[expert_name]

    def unload_all(self):
        print("Unloading all models...")
        self.experts = {}
        self.current_model = None
        torch.cuda.empty_cache()
        print("All models unloaded and cache cleared.")