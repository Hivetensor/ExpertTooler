import unittest
from unittest.mock import patch, MagicMock
import pandas as pd

# Mock the necessary modules
MOCK_MODULES = {
    "datasets": MagicMock(),
    "smart_manager": MagicMock(),
    "expert_tools": MagicMock(),
    "orchestrator": MagicMock(),
    "pandas": MagicMock(),
}

with patch.dict("sys.modules", MOCK_MODULES):
    from mmlu_runner import format_question, evaluate_mmlu_subset

class TestMMLURunner(unittest.TestCase):

    def test_format_question(self):
        """Test that the MMLU question is formatted correctly."""
        row = {
            "question": "What is 2+2?",
            "choices": ["3", "4", "5", "6"],
            "answer": 1,
        }
        expected_output = "Question: What is 2+2?\nChoices:\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
        self.assertEqual(format_question(row), expected_output)

    def test_evaluate_mmlu_subset(self):
        """Test the evaluation loop for a subset of MMLU questions."""
        # Mock agent and dataset
        mock_agent = MagicMock()
        mock_agent.run.side_effect = ["B", "A"]  # Agent returns 'B' then 'A'
        
        mock_dataset = [
            {"question": "Q1", "choices": ["A", "B", "C", "D"], "answer": 1},
            {"question": "Q2", "choices": ["A", "B", "C", "D"], "answer": 2},
        ]

        # Call the function  
        results_df = evaluate_mmlu_subset(mock_agent, mock_dataset, num_questions=2)

        # Assertions
        self.assertEqual(mock_agent.run.call_count, 2)
        self.assertEqual(len(results_df), 2)
        
        # Check first result (correct)
        self.assertTrue(results_df.iloc[0]["is_correct"])
        
        # Check second result (incorrect)
        self.assertFalse(results_df.iloc[1]["is_correct"])

if __name__ == "__main__":
    unittest.main()
