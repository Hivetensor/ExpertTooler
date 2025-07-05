import unittest
from unittest.mock import patch, MagicMock
import torch

# Mock the transformers and langchain libraries
MOCK_MODULES = {
    "torch": MagicMock(),
    "transformers": MagicMock(),
    "langchain.llms": MagicMock(),
    "langchain.agents": MagicMock(),
    "langchain.memory": MagicMock(),
    "datasets": MagicMock(),
    "bitsandbytes": MagicMock(),
}

# Since we are in a testing environment, we don't want to import the actual libraries
# but rather mock them to avoid heavy dependencies and actual model loading.
with patch.dict("sys.modules", MOCK_MODULES):
    from smart_manager import SmartModelManager

class TestSmartModelManager(unittest.TestCase):

    def setUp(self):
        self.expert_configs = {
            "math": {
                "model_id": "mock-math-model",
                "load_in_4bit": True,
            },
            "bio": {
                "model_id": "mock-bio-model",
                "load_in_4bit": False,
            }
        }
        # We need to mock torch.float16 as it is used in the code
        torch.float16 = "float16"


    @patch("smart_manager.AutoModelForCausalLM")
    @patch("smart_manager.AutoTokenizer")
    @patch("smart_manager.HuggingFacePipeline")
    @patch("smart_manager.BitsAndBytesConfig")
    def test_get_expert_loads_model(self, mock_bitsandbytes, mock_hf_pipeline, mock_tokenizer, mock_model):
        """Test that get_expert loads a model that is not cached."""
        manager = SmartModelManager(self.expert_configs)
        
        # Configure mocks
        mock_model.from_pretrained.return_value = "mock_hf_model"
        mock_tokenizer.from_pretrained.return_value = "mock_hf_tokenizer"
        mock_hf_pipeline.from_model_and_tokenizer.return_value = "mock_pipeline"

        # Call the method
        expert_pipeline = manager.get_expert("math")

        # Assertions
        self.assertEqual(expert_pipeline, "mock_pipeline")
        self.assertIn("math", manager.experts)
        self.assertEqual(manager.current_model, "math")
        mock_model.from_pretrained.assert_called_once()
        mock_tokenizer.from_pretrained.assert_called_once()

    @patch("smart_manager.torch.cuda.empty_cache")
    def test_unload_all(self, mock_empty_cache):
        """Test that unload_all clears the experts and cache."""
        manager = SmartModelManager(self.expert_configs)
        manager.experts["math"] = "mock_pipeline"
        manager.current_model = "math"

        manager.unload_all()

        self.assertEqual(manager.experts, {})
        self.assertIsNone(manager.current_model)
        mock_empty_cache.assert_called_once()

if __name__ == "__main__":
    unittest.main()
