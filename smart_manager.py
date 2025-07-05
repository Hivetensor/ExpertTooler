import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain.llms import HuggingFacePipeline

class SmartModelManager:
    def __init__(self, expert_configs):
        self.experts = {}
        self.current_model = None
        self.configs = expert_configs
        self.tokenizer_cache = {}

    def get_expert(self, expert_name):
        if self.current_model and self.current_model != expert_name:
            self.unload_all()

        if expert_name not in self.experts:
            print(f"Loading expert: {expert_name}...")
            config = self.configs[expert_name]
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=config.get("load_in_4bit", True),
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )

            model = AutoModelForCausalLM.from_pretrained(
                config["model_id"],
                quantization_config=quantization_config,
                device_map="auto",
            )
            
            if config["model_id"] not in self.tokenizer_cache:
                self.tokenizer_cache[config["model_id"]] = AutoTokenizer.from_pretrained(config["model_id"])
            
            tokenizer = self.tokenizer_cache[config["model_id"]]

            pipeline = HuggingFacePipeline.from_model_and_tokenizer(
                model=model,
                tokenizer=tokenizer,
                task="text-generation",
                model_kwargs={"max_length": config.get("max_length", 1024)},
            )
            self.experts[expert_name] = pipeline
            self.current_model = expert_name
            print(f"Expert {expert_name} loaded.")

        return self.experts[expert_name]

    def unload_all(self):
        print("Unloading all models...")
        self.experts = {}
        self.current_model = None
        torch.cuda.empty_cache()
        print("All models unloaded and cache cleared.")
