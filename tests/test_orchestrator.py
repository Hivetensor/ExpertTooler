import unittest
from unittest.mock import patch, MagicMock
import torch

# Mock the necessary modules
MOCK_MODULES = {
    "torch": MagicMock(),
    "transformers": MagicMock(),
    "langchain.llms": MagicMock(),
    "langchain.agents": MagicMock(),
    "langchain.memory": MagicMock(),
    "bitsandbytes": MagicMock(),
}

with patch.dict("sys.modules", MOCK_MODULES):
    from orchestrator import create_orchestrator_agent

class TestOrchestrator(unittest.TestCase):

    def setUp(self):
        self.base_model_config = {
            "model_id": "mock-base-model",
            "max_length": 1024,
            "load_in_4bit": True,
        }
        self.mock_tools = [MagicMock(), MagicMock()]
        # We need to mock torch.float16 as it is used in the code
        torch.float16 = "float16"

    @patch("orchestrator.AutoModelForCausalLM")
    @patch("orchestrator.AutoTokenizer")
    @patch("orchestrator.HuggingFacePipeline")
    @patch("orchestrator.initialize_agent")
    @patch("orchestrator.ConversationBufferMemory")
    @patch("orchestrator.BitsAndBytesConfig")
    def test_create_orchestrator_agent(self, mock_bitsandbytes, mock_memory, mock_init_agent, mock_hf_pipeline, mock_tokenizer, mock_model):
        """Test that the orchestrator agent is created and configured correctly."""
        # Configure mocks
        mock_model.from_pretrained.return_value = "mock_hf_model"
        mock_tokenizer.from_pretrained.return_value = "mock_hf_tokenizer"
        mock_hf_pipeline.from_model_and_tokenizer.return_value = "mock_pipeline"
        mock_init_agent.return_value = "mock_agent"
        mock_memory.return_value = "mock_memory"

        # Call the function
        agent = create_orchestrator_agent(self.mock_tools, self.base_model_config)

        # Assertions
        self.assertEqual(agent, "mock_agent")
        mock_model.from_pretrained.assert_called_once_with(
            self.base_model_config["model_id"],
            quantization_config=unittest.mock.ANY,
            device_map="auto",
        )
        mock_init_agent.assert_called_once_with(
            tools=self.mock_tools,
            llm="mock_pipeline",
            agent=MOCK_MODULES["langchain.agents"].AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory="mock_memory",
            handle_parsing_errors=True,
        )

if __name__ == "__main__":
    unittest.main()
