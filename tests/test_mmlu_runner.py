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
            "choice0": "3",
            "choice1": "4",
            "choice2": "5",
            "choice3": "6",
            "answer": "B",
        }
        expected_output = "Question: What is 2+2?\nChoices:\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:"
        self.assertEqual(format_question(row), expected_output)

    def test_evaluate_mmlu_subset(self):
        """Test the evaluation loop for a subset of MMLU questions."""
        # Mock agent and dataset
        mock_agent = MagicMock()
        mock_agent.run.side_effect = ["B", "A"]  # Agent returns 'B' then 'A'
        
        mock_dataset = [
            {"question": "Q1", "choice0": "A", "choice1": "B", "choice2": "C", "choice3": "D", "answer": "B"},
            {"question": "Q2", "choice0": "A", "choice1": "B", "choice2": "C", "choice3": "D", "answer": "C"},
        ]

        # Call the function
        with patch('mmlu_runner.pd.DataFrame', new=pd.DataFrame) as mock_df:
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
