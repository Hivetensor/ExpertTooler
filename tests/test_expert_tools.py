import unittest
from unittest.mock import MagicMock, patch

# Mock the necessary modules before import
MOCK_MODULES = {
    "langchain.tools": MagicMock(),
    "smart_manager": MagicMock(),
}

with patch.dict("sys.modules", MOCK_MODULES):
    from expert_tools import create_expert_tools
    from smart_manager import SmartModelManager

class TestExpertTools(unittest.TestCase):

    def test_create_expert_tools(self):
        """Test that expert tools are created correctly."""
        # Setup mock SmartModelManager
        mock_manager = MagicMock(spec=SmartModelManager)
        mock_manager.configs = {
            "math": {
                "name": "MathematicsExpert",
                "description": "Math expert description.",
            },
            "bio": {
                "name": "BiologyExpert",
                "description": "Bio expert description.",
            }
        }
        
        # Mock the Tool class from langchain
        mock_tool_class = MOCK_MODULES["langchain.tools"].Tool
        
        # Call the function
        tools = create_expert_tools(mock_manager)

        # Assertions
        self.assertEqual(len(tools), 2)
        
        # Check the first tool
        mock_tool_class.assert_any_call(
            name="MathematicsExpert",
            description="Math expert description.",
            func=unittest.mock.ANY,
        )
        
        # Check the second tool
        mock_tool_class.assert_any_call(
            name="BiologyExpert",
            description="Bio expert description.",
            func=unittest.mock.ANY,
        )

if __name__ == "__main__":
    unittest.main()
