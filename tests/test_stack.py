import ast
import sys
from pathlib import Path

import unittest
from unittest.mock import patch, MagicMock


# Add the src directory to the Python path
sys.path.append(str(Path(__file__).parent.parent))


from src.data.stack import extract_functions_with_docstrings, create_docstring_tasks, is_quality_docstring, is_quality_code


class TestQualityHeuristics(unittest.TestCase):
    def test_property_getter_rejected(self):
        """Test that property getters are rejected."""
        code = (
            "class SomeClass:\n"
            "    def __init__(self):\n"
            "        self._value = None\n"
            "        \n"
            "    @property\n"
            "    def value(self):\n"
            "        \"\"\"Gets the value of this SomeClass.\n"
            "        \n"
            "        This is a sufficiently long docstring that would normally pass,\n"
            "        but since it's a property getter it should be rejected.\n"
            "        \n"
            "        :return: The value of this SomeClass.\n"
            "        :rtype: str\n"
            "        \"\"\"\n"
            "        return self._value\n"
        )
        
        # Parse the AST to get the node
        tree = ast.parse(code)
        property_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == 'value':
                property_node = node
                break
                
        self.assertIsNotNone(property_node, "Failed to find property node in AST")
        
        # Get the docstring
        docstring = ast.get_docstring(property_node)
        
        # Should be rejected due to being a property
        self.assertFalse(is_quality_docstring(docstring, node=property_node, code=code, min_length=10))
        
        # Using extract_functions_with_docstrings should also reject it
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        self.assertEqual(0, len(functions), "Property getter should be rejected")
        
    def test_simple_function_rejected(self):
        """Test that simple one-line functions are rejected."""
        code = (
            "def get_user_id(user):\n"
            "    \"\"\"\n"
            "    Get the ID of the user object.\n"
            "    \n"
            "    This is a very simple function that just returns a property\n"
            "    of the user object. Even though the docstring is substantial,\n"
            "    the function itself is trivial.\n"
            "    \n"
            "    Args:\n"
            "        user: The user object\n"
            "        \n"
            "    Returns:\n"
            "        The user ID\n"
            "    \"\"\"\n"
            "    return user.id\n"
        )
        
        # Parse the AST to get the node
        tree = ast.parse(code)
        function_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_node = node
                break
                
        self.assertIsNotNone(function_node, "Failed to find function node in AST")
        
        # Get the docstring
        docstring = ast.get_docstring(function_node)
        
        # Should be rejected due to being a simple function
        self.assertFalse(is_quality_docstring(docstring, node=function_node, code=code, min_length=10))
        
        # Using extract_functions_with_docstrings should also reject it
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        self.assertEqual(0, len(functions), "Simple function should be rejected")
        
    def test_boilerplate_docstring_rejected(self):
        """Test that docstrings with boilerplate indicators are rejected."""
        # Docstring with noqa comment
        docstring1 = (
            "Gets the return_code of this Response.  # noqa: E501\n"
            "\n"
            "This is a substantial docstring with good content\n"
            "that would normally pass the quality check.\n"
            "\n"
            ":return: The return_code of this Response.\n"
            ":rtype: str"
        )
        
        # Docstring with "Gets the" pattern
        docstring2 = (
            "Gets the username from the request object.\n"
            "\n"
            "This docstring looks like it might be auto-generated\n"
            "or follow some standard template. We want to filter these out.\n"
            "\n"
            "Args:\n"
            "    request: The request object\n"
            "\n"
            "Returns:\n"
            "    The username"
        )
        
        # Both should be rejected
        self.assertFalse(is_quality_docstring(docstring1, min_length=10))
        self.assertFalse(is_quality_docstring(docstring2, min_length=10))
        
    def test_complex_function_accepted(self):
        """Test that a complex function with good docstring is accepted."""
        code = (
            "def process_user_data(users, fields=None, include_metadata=False):\n"
            "    \"\"\"\n"
            "    Process a list of users and extract relevant fields.\n"
            "    \n"
            "    This function takes a list of user objects and processes them\n"
            "    to extract the specified fields. It can optionally include metadata.\n"
            "    \n"
            "    Args:\n"
            "        users: List of user objects to process\n"
            "        fields: List of fields to extract (default: all fields)\n"
            "        include_metadata: Whether to include metadata (default: False)\n"
            "        \n"
            "    Returns:\n"
            "        List of processed user data\n"
            "    \"\"\"\n"
            "    if fields is None:\n"
            "        fields = ['id', 'name', 'email']\n"
            "        \n"
            "    result = []\n"
            "    for user in users:\n"
            "        user_data = {}\n"
            "        for field in fields:\n"
            "            if hasattr(user, field):\n"
            "                user_data[field] = getattr(user, field)\n"
            "                \n"
            "        if include_metadata and hasattr(user, 'metadata'):\n"
            "            user_data['metadata'] = user.metadata\n"
            "            \n"
            "        result.append(user_data)\n"
            "        \n"
            "    return result\n"
        )
        
        # Parse the AST to get the node
        tree = ast.parse(code)
        function_node = None
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_node = node
                break
                
        self.assertIsNotNone(function_node, "Failed to find function node in AST")
        
        # Get the docstring
        docstring = ast.get_docstring(function_node)
        
        # Should be accepted as it's a complex function with good docstring
        self.assertTrue(is_quality_docstring(docstring, node=function_node, code=code, min_length=10))
        
        # Using extract_functions_with_docstrings should also accept it
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        self.assertEqual(1, len(functions), "Complex function should be accepted")
        self.assertEqual("process_user_data", functions[0]["name"])


class TestExtractFunctions(unittest.TestCase):
    def test_extract_simple_function(self):
        # Simple function with docstring, but with enough complexity to pass quality checks
        code = (
            "\n"
            "def example_function(param1, param2):\n"
            "    \"\"\"\n"
            "    This is a good docstring with sufficient length to meet our criteria.\n"
            "    It describes what the function does in detail.\n"
            "    \n"
            "    Args:\n"
            "        param1: Description of param1\n"
            "        param2: Description of param2\n"
            "        \n"
            "    Returns:\n"
            "        Description of return value\n"
            "    \"\"\"\n"
            "    # Implementation with enough statements to pass quality check\n"
            "    if param1 > param2:\n"
            "        result = param1 + param2\n"
            "    else:\n"
            "        result = param1 * param2\n"
            "        \n"
            "    # Additional operations\n"
            "    for i in range(param1):\n"
            "        result += i\n"
            "        \n"
            "    return result\n"
        )
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        
        self.assertEqual(1, len(functions))
        self.assertEqual("example_function", functions[0]["name"])
        self.assertIn("This is a good docstring", functions[0]["docstring"])
        self.assertIn("def example_function", functions[0]["implementation"])
        self.assertIn("return result", functions[0]["implementation"])
        
        # Check masked code - function definition should not be present
        self.assertNotIn("def example_function", functions[0]["masked_code"])
        # But the placeholder comment should include the function name
        self.assertIn("MASKED: example_function function", functions[0]["masked_code"])
        
    def test_extract_nested_function(self):
        # Function with nested function
        code = (
            "\n"
            "def outer_function(x, y):\n"
            "    \"\"\"\n"
            "    This is a good docstring for the outer function with sufficient length.\n"
            "    It has multiple paragraphs and describes params.\n"
            "    \n"
            "    Args:\n"
            "        x: First parameter\n"
            "        y: Second parameter\n"
            "        \n"
            "    Returns:\n"
            "        The result of inner calculation\n"
            "    \"\"\"\n"
            "    # Complex enough implementation to pass quality checks\n"
            "    if x < 0 or y < 0:\n"
            "        raise ValueError(\"Input must be positive\")\n"
            "        \n"
            "    def inner_function(a, b):\n"
            "        \"\"\"Inner function docstring.\"\"\"\n"
            "        # More complex implementation\n"
            "        if a > b:\n"
            "            return a * b\n"
            "        else:\n"
            "            return a + b\n"
            "    \n"
            "    # Process with the nested function\n"
            "    result = inner_function(x, y)\n"
            "    \n"
            "    # Additional processing\n"
            "    for i in range(min(x, 5)):\n"
            "        result += inner_function(i, y)\n"
            "        \n"
            "    return result\n"
        )
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        
        # Should extract outer function only (inner has too short docstring)
        self.assertEqual(1, len(functions))
        self.assertEqual("outer_function", functions[0]["name"])
        
        # Check that entire implementation including nested function is captured
        self.assertIn("def inner_function", functions[0]["implementation"])
        self.assertIn("return result", functions[0]["implementation"])
        
        # Check masked code - function definition should not be present
        self.assertNotIn("def outer_function", functions[0]["masked_code"])
        # But the placeholder comment should include the function name
        self.assertIn("MASKED: outer_function function", functions[0]["masked_code"])
        
    def test_extract_multiline_function(self):
        # Function with complex body and decorators
        code = (
            "\n"
            "@decorator1\n"
            "@decorator2\n"
            "def complex_function(\n"
            "    param1,\n"
            "    param2,\n"
            "    param3=None\n"
            "):\n"
            "    \"\"\"\n"
            "    This is a complex function with decorators and multi-line signature.\n"
            "    It demonstrates that we need to capture everything correctly.\n"
            "    \n"
            "    Args:\n"
            "        param1: First parameter\n"
            "        param2: Second parameter\n"
            "        param3: Optional parameter\n"
            "        \n"
            "    Returns:\n"
            "        Complex result\n"
            "    \"\"\"\n"
            "    # Complex implementation\n"
            "    if param3 is None:\n"
            "        param3 = {}\n"
            "    \n"
            "    # More code\n"
            "    for item in param1:\n"
            "        if item in param2:\n"
            "            param3[item] = param2[item]\n"
            "    \n"
            "    # Final calculation\n"
            "    return sum(param3.values())\n"
        )
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        
        self.assertEqual(1, len(functions))
        self.assertEqual("complex_function", functions[0]["name"])
        
        # Check that decorators are included
        self.assertIn("@decorator1", functions[0]["implementation"])
        self.assertIn("@decorator2", functions[0]["implementation"])
        
        # Check that entire implementation is captured
        self.assertIn("return sum(param3.values())", functions[0]["implementation"])
        
        # Check masked code - function definition should not be present
        self.assertNotIn("def complex_function", functions[0]["masked_code"])
        # But the placeholder comment should include the function name
        self.assertIn("MASKED: complex_function function", functions[0]["masked_code"])
        
    def test_extract_class_method(self):
        # Class with method
        code = (
            "\n"
            "class ExampleClass:\n"
            "    def __init__(self, value):\n"
            "        self.value = value\n"
            "        \n"
            "    def example_method(self, param):\n"
            "        \"\"\"\n"
            "        This is a method with a good docstring of sufficient length.\n"
            "        It explains what the method does in detail.\n"
            "        \n"
            "        Args:\n"
            "            param: Input parameter\n"
            "            \n"
            "        Returns:\n"
            "            Calculated result\n"
            "        \"\"\"\n"
            "        # Method implementation with enough complexity to pass quality check\n"
            "        result = self.value * param\n"
            "        \n"
            "        # Additional processing\n"
            "        if result > 100:\n"
            "            result = result / 2\n"
            "        \n"
            "        # More operations\n"
            "        for i in range(param):\n"
            "            if i % 2 == 0:\n"
            "                result += i\n"
            "        \n"
            "        return result\n"
        )
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        
        self.assertEqual(1, len(functions))
        self.assertEqual("example_method", functions[0]["name"])
        
        # Check that method implementation is correct
        self.assertIn("def example_method", functions[0]["implementation"])
        self.assertIn("return result", functions[0]["implementation"])
        
        # Check masked code - function definition should not be present
        self.assertNotIn("def example_method", functions[0]["masked_code"])
        # But the placeholder comment should include the function name
        self.assertIn("MASKED: example_method function", functions[0]["masked_code"])
        
        # But class and init should remain
        self.assertIn("class ExampleClass", functions[0]["masked_code"])
        self.assertIn("def __init__", functions[0]["masked_code"])
        
    def test_no_implementation_lines_lost(self):
        """Test that no implementation lines are lost, especially closing braces or indented blocks"""
        code = (
            "\n"
            "def function_with_many_blocks(x):\n"
            "    \"\"\"\n"
            "    This is a function with many nested blocks and complex structure.\n"
            "    We want to make sure all closing braces and code blocks are captured.\n"
            "    \n"
            "    Args:\n"
            "        x: Input parameter\n"
            "        \n"
            "    Returns:\n"
            "        Processed result\n"
            "    \"\"\"\n"
            "    # Complex implementation with multiple branches\n"
            "    result = 0\n"
            "    \n"
            "    # First branch of logic\n"
            "    if x > 0:\n"
            "        if x > 10:\n"
            "            result = x * 2\n"
            "        else:\n"
            "            result = x + 2\n"
            "    else:\n"
            "        # Second branch handles negative numbers\n"
            "        for i in range(abs(x)):\n"
            "            if i % 2 == 0:\n"
            "                continue\n"
            "            result = -x\n"
            "            break\n"
            "        else:\n"
            "            result = 0\n"
            "    \n"
            "    # Additional processing based on result\n"
            "    if result > 20:\n"
            "        for i in range(min(5, result)):\n"
            "            result += i\n"
            "    \n"
            "    # Final lines with different indentation\n"
            "    return result\n"
        )
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        
        self.assertEqual(1, len(functions))
        
        # Check full implementation is captured
        impl = functions[0]["implementation"]
        
        # Count indentation levels to ensure structure is preserved
        for line in ["if x > 0:", "if x > 10:", "result = x * 2", "else:", "return result"]:
            self.assertIn(line, impl)
            
        # Check masked code - function definition should not be present
        self.assertNotIn("def function_with_many_blocks", functions[0]["masked_code"])
        # But the placeholder comment should include the function name
        self.assertIn("MASKED: function_with_many_blocks function", functions[0]["masked_code"])


class TestCreateDocstringTasks(unittest.TestCase):
    @patch('src.data.stack.load_dataset')
    @patch('src.data.stack.is_quality_code')
    def test_create_docstring_tasks(self, mock_is_quality_code, mock_datasets):
        # Mock the code quality check to always pass
        mock_is_quality_code.return_value = (True, "Passed quality checks")
        
        # Mock the dataset response
        mock_dataset = MagicMock()
        mock_datasets.return_value = mock_dataset
        
        # Create sample data with a complex function that will pass quality checks
        sample1 = {"content": (
            "\n"
            "def good_complex_function(x, y, z=None):\n"
            "    \"\"\"\n"
            "    This is a good docstring with sufficient length to meet our criteria.\n"
            "    It has multiple lines and describes parameters.\n"
            "    \n"
            "    Args:\n"
            "        x: First parameter\n"
            "        y: Second parameter\n"
            "        z: Optional parameter\n"
            "        \n"
            "    Returns:\n"
            "        The processed result\n"
            "    \"\"\"\n"
            "    # This is a more complex function with multiple statements\n"
            "    if z is None:\n"
            "        z = {}\n"
            "        \n"
            "    result = x + y\n"
            "    \n"
            "    # Process additional data\n"
            "    for i in range(result):\n"
            "        z[i] = i * result\n"
            "        \n"
            "    return z\n"
        )}
        
        # A simple function that should be rejected
        sample2 = {"content": (
            "\n"
            "def bad_function(x):\n"
            "    \"\"\"Too short.\"\"\"\n"
            "    return x * 2\n"
        )}
        
        # Set up the mock dataset to return our samples
        mock_dataset.__iter__.return_value = [sample1, sample2]
        
        # Call the function
        tasks = create_docstring_tasks(max_samples=2, min_docstring_length=10)
        
        # Verify results
        self.assertEqual(1, len(tasks))  # Only one function with good docstring
        self.assertEqual("good_complex_function", tasks[0].function_name)
        self.assertIn("Args:", tasks[0].docstring)
        self.assertIn("return z", tasks[0].implementation)
        self.assertNotIn("return z", tasks[0].masked_code)
        
        # Check that the function was called with correct parameters
        mock_datasets.assert_called_once_with(
            "bigcode/the-stack-smol", 
            data_dir="data/python", 
            split="train[:2]"
        )


class TestCodeQuality(unittest.TestCase):
    def test_sql_detection(self):
        """Test detection of SQL queries in code."""
        code = (
            "def fetch_users(db_conn):\n"
            "    \"\"\"Fetch all users from the database.\n"
            "    \n"
            "    Args:\n"
            "        db_conn: Database connection\n"
            "        \n"
            "    Returns:\n"
            "        List of users\n"
            "    \"\"\"\n"
            "    query = \"SELECT * FROM users WHERE active = True ORDER BY created_at DESC\"\n"
            "    cursor = db_conn.cursor()\n"
            "    cursor.execute(query)\n"
            "    return cursor.fetchall()\n"
        )
        
        # Parse the AST
        tree = ast.parse(code)
        function_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Should be rejected due to containing SQL
        is_quality, reason = is_quality_code(function_node, code)
        self.assertFalse(is_quality)
        self.assertIn("SQL", reason)
        
    def test_too_many_variables(self):
        """Test detection of functions with too many variables."""
        # Function with many variable assignments
        var_declarations = "\n    ".join([f"var{i} = {i}" for i in range(20)])
        code = (
            "def too_many_vars():\n"
            "    \"\"\"Function with too many variables.\n"
            "    \n"
            "    This function defines many variables and should be rejected.\n"
            "    \"\"\"\n"
            f"    {var_declarations}\n"
            "    return var1 + var2\n"
        )
        
        # Parse the AST
        tree = ast.parse(code)
        function_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Should be rejected due to too many variables
        is_quality, reason = is_quality_code(function_node, code)
        self.assertFalse(is_quality)
        self.assertIn("Too many variables", reason)
        
    def test_too_simple_function(self):
        """Test detection of functions that are too simple."""
        code = (
            "def simple_function(x):\n"
            "    \"\"\"A very simple function.\n"
            "    \n"
            "    This function is too simple and should be rejected.\n"
            "    \n"
            "    Args:\n"
            "        x: Input value\n"
            "        \n"
            "    Returns:\n"
            "        Computed result\n"
            "    \"\"\"\n"
            "    return x * 2\n"
        )
        
        # Parse the AST
        tree = ast.parse(code)
        function_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Should be rejected due to being too simple
        is_quality, reason = is_quality_code(function_node, code)
        self.assertFalse(is_quality)
        self.assertIn("Too simple", reason)
        
    def test_quality_code_accepted(self):
        """Test that quality code is accepted."""
        code = (
            "def process_items(items, threshold=0.5):\n"
            "    \"\"\"Process a list of items and filter based on threshold.\n"
            "    \n"
            "    This function has good structure and reasonable complexity.\n"
            "    \n"
            "    Args:\n"
            "        items: List of items to process\n"
            "        threshold: Filter threshold (default: 0.5)\n"
            "        \n"
            "    Returns:\n"
            "        Processed results\n"
            "    \"\"\"\n"
            "    results = []\n"
            "    for item in items:\n"
            "        score = calculate_score(item)\n"
            "        if score > threshold:\n"
            "            processed = transform_item(item)\n"
            "            results.append(processed)\n"
            "        else:\n"
            "            # Skip low-scoring items\n"
            "            continue\n"
            "            \n"
            "    # Apply final processing\n"
            "    return sorted(results, key=lambda x: x.priority)\n"
        )
        
        # Parse the AST
        tree = ast.parse(code)
        function_node = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        
        # Should be accepted as it has good structure
        is_quality, reason = is_quality_code(function_node, code)
        self.assertTrue(is_quality)
        self.assertIn("Passed", reason)
        
    def test_extract_functions_filters_quality(self):
        """Test that extract_functions_with_docstrings filters by code quality."""
        # Create a file with multiple functions of varying quality
        code = (
            "def sql_function(db):\n"
            "    \"\"\"Function with SQL query.\n"
            "    \n"
            "    This has a good docstring but uses SQL.\n"
            "    \n"
            "    Args:\n"
            "        db: Database connection\n"
            "        \n"
            "    Returns:\n"
            "        Query results\n"
            "    \"\"\"\n"
            "    return db.execute('SELECT * FROM table')\n"
            "\n"
            "def good_function(data):\n"
            "    \"\"\"Process data with good algorithm.\n"
            "    \n"
            "    This function has good docstring and good code.\n"
            "    \n"
            "    Args:\n"
            "        data: Input data\n"
            "        \n"
            "    Returns:\n"
            "        Processed data\n"
            "    \"\"\"\n"
            "    result = []\n"
            "    for item in data:\n"
            "        if item.is_valid():\n"
            "            transformed = item.transform()\n"
            "            result.append(transformed)\n"
            "    return result\n"
        )
        
        # Should extract only the good_function
        functions = extract_functions_with_docstrings(code, min_docstring_length=10)
        
        # Verify only good_function was extracted
        self.assertEqual(1, len(functions))
        self.assertEqual("good_function", functions[0]["name"])


if __name__ == "__main__":
    unittest.main() 