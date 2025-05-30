import unittest
from src.rewards.diff import split_diff_by_files, normalize_file_diffs, unified_diff_similarity_reward_func, unified_diff_file_match_reward_func


class TestUnifiedDiffRewards(unittest.TestCase):
    
    def test_split_diff_by_files(self):
        """Test splitting a unified diff into individual file diffs."""
        diff_text = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
 def hello():
-    print("hello")
+    print("hello world")

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -5,7 +5,7 @@
 def goodbye():
-    print("bye")
+    print("goodbye")
"""
        files = split_diff_by_files(diff_text)
        self.assertEqual(len(files), 2)
        self.assertTrue(files[0].startswith("diff --git a/file1.py"))
        self.assertTrue(files[1].startswith("diff --git a/file2.py"))
    
    def test_normalize_file_diffs(self):
        """Test extracting filenames and changes from file diffs."""
        file_diffs = ["""diff --git a/src/example.py b/src/example.py
--- a/src/example.py
+++ b/src/example.py
@@ -1,3 +1,3 @@
 def hello():
-    print("hello")
+    print("hello world")
 return None"""]
        
        normalized = normalize_file_diffs(file_diffs)
        self.assertIn("src/example.py", normalized)
        changes = normalized["src/example.py"]
        self.assertEqual(len(changes), 2)
        self.assertIn('-    print("hello")', changes)
        self.assertIn('+    print("hello world")', changes)
    
    def test_unified_diff_similarity_same_order(self):
        """Test that identical diffs in same order get perfect score."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    another old
+    another new"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff1])
        self.assertEqual(scores[0], 1.0)
    
    def test_unified_diff_similarity_different_order(self):
        """Test that identical diffs in different order still get perfect score."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    another old
+    another new"""
        
        diff2 = """diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    another old
+    another new

diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        self.assertEqual(scores[0], 1.0)
    
    def test_unified_diff_similarity_partial_match(self):
        """Test partial match when one file is missing."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    another old
+    another new"""
        
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        # Should be 0.5 since 1 out of 2 files match
        self.assertEqual(scores[0], 0.5)
    
    def test_unified_diff_similarity_no_match(self):
        """Test completely different diffs."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line"""
        
        diff2 = """diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    different old
+    different new"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        self.assertEqual(scores[0], 0.0)
    
    def test_unified_diff_similarity_empty_generated(self):
        """Test when generated diff is empty string."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [""])
        self.assertEqual(scores[0], 0.0)
    
    def test_unified_diff_similarity_batch(self):
        """Test batch processing of multiple diffs."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        diff2 = """diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old2
+    new2"""
        
        # First pair: identical
        # Second pair: different files
        scores = unified_diff_similarity_reward_func([diff1, diff1], ["", ""], [diff1, diff2])
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1], 0.0)


    def test_unified_diff_similarity_similar_changes(self):
        """Test that similar changes get high scores."""
        # Original has 3 changes
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,5 +1,5 @@
-    def foo():
+    def bar():
         pass
-    x = 1
+    x = 2
-    y = 3
+    y = 4"""
        
        # Generated has 2 out of 3 same changes
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,5 +1,5 @@
-    def foo():
+    def bar():
         pass
-    x = 1
+    x = 2"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        # Should get a high score since most changes match
        self.assertGreater(scores[0], 0.6)
        self.assertLess(scores[0], 1.0)
    
    def test_unified_diff_similarity_dissimilar_changes(self):
        """Test that dissimilar changes get low scores."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    print("hello")
+    print("hello world")"""
        
        # Completely different changes
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -10,3 +10,3 @@
-    x = 1
+    x = 2"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        # Should get zero since changes are completely different
        self.assertEqual(scores[0], 0.0)
    
    def test_unified_diff_similarity_multiple_files_partial_match(self):
        """Test multiple files with partial matches."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old1
+    new1

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old2
+    new2

diff --git a/file3.py b/file3.py
--- a/file3.py
+++ b/file3.py
@@ -1,3 +1,3 @@
-    old3
+    new3"""
        
        # Only has 2 out of 3 files, but changes match
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old1
+    new1

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old2
+    new2"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        # Should be 2/3 since 2 out of 3 files match perfectly
        self.assertAlmostEqual(scores[0], 2/3, places=2)
    
    def test_unified_diff_similarity_whitespace_differences(self):
        """Test that whitespace differences affect similarity."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    def foo():
+    def bar():"""
        
        # Same change but with extra spaces
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    def foo():
+    def  bar():"""  # Note extra space
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        # SequenceMatcher gives partial match for similar strings with whitespace diff
        self.assertGreater(scores[0], 0.4)
        self.assertLess(scores[0], 0.7)
    
    def test_unified_diff_similarity_line_number_differences(self):
        """Test that same changes at different line numbers still match."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -10,3 +10,3 @@
-    old line
+    new line"""
        
        # Same change but at different line numbers
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -20,3 +20,3 @@
-    old line
+    new line"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        # Should be perfect match since changes are identical
        self.assertEqual(scores[0], 1.0)
    
    def test_unified_diff_similarity_context_differences(self):
        """Test that different context lines don't affect change matching."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,5 +1,5 @@
 def function():
     # context line 1
-    old code
+    new code
     # context line 2"""
        
        # Same change but with different context
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,5 +1,5 @@
 def other_function():
     # different context
-    old code
+    new code
     # another context"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""], [diff2])
        # Should be perfect since we only compare +/- lines
        self.assertEqual(scores[0], 1.0)
    
    def test_unified_diff_similarity_with_test_patch(self):
        """Test when both patch and test_patch are provided."""
        patch = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True"""
        
        test_patch = """diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,3 @@
-    assert result == False
+    assert result == True"""
        
        # Generated diff should contain both changes
        generated_diff = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True

diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,3 @@
-    assert result == False
+    assert result == True"""
        
        scores = unified_diff_similarity_reward_func([patch], [test_patch], [generated_diff])
        self.assertEqual(scores[0], 1.0)
    
    def test_unified_diff_similarity_partial_test_patch(self):
        """Test when generated diff only contains patch but not test_patch."""
        patch = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True"""
        
        test_patch = """diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,3 @@
-    assert result == False
+    assert result == True"""
        
        # Generated diff only contains the main patch
        generated_diff = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True"""
        
        scores = unified_diff_similarity_reward_func([patch], [test_patch], [generated_diff])
        # Should be 0.5 since only 1 out of 2 files match
        self.assertEqual(scores[0], 0.5)
    
    def test_unified_diff_similarity_reversed_patch_order(self):
        """Test when generated diff has patch and test_patch in reversed order."""
        patch = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True"""
        
        test_patch = """diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,3 @@
-    assert result == False
+    assert result == True"""
        
        # Generated diff has test changes first, then main changes
        generated_diff = """diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,3 @@
-    assert result == False
+    assert result == True

diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True"""
        
        scores = unified_diff_similarity_reward_func([patch], [test_patch], [generated_diff])
        # Should be 1.0 since both files match regardless of order
        self.assertEqual(scores[0], 1.0)
    
    def test_unified_diff_similarity_extra_files_in_generated(self):
        """Test when generated diff contains extra files beyond patch and test_patch."""
        patch = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True"""
        
        test_patch = """diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,3 @@
-    assert result == False
+    assert result == True"""
        
        # Generated diff has patch, test_patch, and an extra file
        generated_diff = """diff --git a/src/main.py b/src/main.py
--- a/src/main.py
+++ b/src/main.py
@@ -1,3 +1,3 @@
-    return False
+    return True

diff --git a/tests/test_main.py b/tests/test_main.py
--- a/tests/test_main.py
+++ b/tests/test_main.py
@@ -1,3 +1,3 @@
-    assert result == False
+    assert result == True

diff --git a/docs/README.md b/docs/README.md
--- a/docs/README.md
+++ b/docs/README.md
@@ -1,3 +1,3 @@
-    Old docs
+    New docs"""
        
        scores = unified_diff_similarity_reward_func([patch], [test_patch], [generated_diff])
        # Should be 2/3 since 2 out of 3 files match the oracle
        self.assertAlmostEqual(scores[0], 2/3, places=2)
    
    def test_unified_diff_similarity_multiple_batches(self):
        """Test batch processing with different patch and test_patch combinations."""
        patches = [
            """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-old_a
+new_a""",
            """diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1,1 +1,1 @@
-old_b
+new_b"""
        ]
        
        test_patches = [
            """diff --git a/test_a.py b/test_a.py
--- a/test_a.py
+++ b/test_a.py
@@ -1,1 +1,1 @@
-test_old_a
+test_new_a""",
            """diff --git a/test_b.py b/test_b.py
--- a/test_b.py
+++ b/test_b.py
@@ -1,1 +1,1 @@
-test_old_b
+test_new_b"""
        ]
        
        generated_diffs = [
            # First: perfect match for both
            """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,1 +1,1 @@
-old_a
+new_a

diff --git a/test_a.py b/test_a.py
--- a/test_a.py
+++ b/test_a.py
@@ -1,1 +1,1 @@
-test_old_a
+test_new_a""",
            # Second: only main patch, missing test patch
            """diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1,1 +1,1 @@
-old_b
+new_b"""
        ]
        
        scores = unified_diff_similarity_reward_func(patches, test_patches, generated_diffs)
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0], 1.0)  # Perfect match
        self.assertEqual(scores[1], 0.5)  # Half match
    
    def test_unified_diff_similarity_realistic_similar_structure(self):
        """Test with realistic diffs that have similar structure but different specifics."""
        # Original patch fixes a bug in error handling
        patch = """diff --git a/src/api/handler.py b/src/api/handler.py
--- a/src/api/handler.py
+++ b/src/api/handler.py
@@ -42,18 +42,22 @@ class APIHandler:
     def process_request(self, request):
         try:
             # Validate request
-            if not request.get('data'):
-                return {'error': 'No data'}
+            if not request or not isinstance(request, dict):
+                raise ValueError('Invalid request format')
+            
+            if 'data' not in request:
+                raise ValueError('Missing required field: data')
             
             # Process the data
             data = request['data']
-            result = self.processor.process(data)
+            validated_data = self.validator.validate(data)
+            result = self.processor.process(validated_data)
             
-            return {'result': result}
+            return {'status': 'success', 'result': result}
             
         except Exception as e:
             logger.error(f'Error processing request: {e}')
-            return {'error': str(e)}
+            return {'status': 'error', 'message': str(e)}"""
        
        # Test patch updates the corresponding test
        test_patch = """diff --git a/tests/test_api_handler.py b/tests/test_api_handler.py
--- a/tests/test_api_handler.py
+++ b/tests/test_api_handler.py
@@ -15,8 +15,12 @@ class TestAPIHandler(unittest.TestCase):
     def test_process_request_missing_data(self):
         request = {}
         response = self.handler.process_request(request)
-        self.assertEqual(response['error'], 'No data')
+        self.assertEqual(response['status'], 'error')
+        self.assertIn('data', response['message'])
+    
+    def test_process_request_invalid_format(self):
+        response = self.handler.process_request(None)
+        self.assertEqual(response['status'], 'error')"""
        
        # Generated diff has similar structure but different implementation details
        generated_diff = """diff --git a/src/api/handler.py b/src/api/handler.py
--- a/src/api/handler.py
+++ b/src/api/handler.py
@@ -42,18 +42,21 @@ class APIHandler:
     def process_request(self, request):
         try:
             # Validate request
-            if not request.get('data'):
-                return {'error': 'No data'}
+            if request is None:
+                raise TypeError('Request cannot be None')
+            
+            if 'data' not in request or not request['data']:
+                raise KeyError('Data field is required')
             
             # Process the data
             data = request['data']
-            result = self.processor.process(data)
+            clean_data = self.sanitizer.clean(data)
+            result = self.processor.handle(clean_data)
             
-            return {'result': result}
+            return {'success': True, 'data': result}
             
         except Exception as e:
             logger.error(f'Error processing request: {e}')
-            return {'error': str(e)}
+            return {'success': False, 'error': str(e)}

diff --git a/tests/test_api_handler.py b/tests/test_api_handler.py
--- a/tests/test_api_handler.py
+++ b/tests/test_api_handler.py
@@ -15,8 +15,11 @@ class TestAPIHandler(unittest.TestCase):
     def test_process_request_missing_data(self):
         request = {}
         response = self.handler.process_request(request)
-        self.assertEqual(response['error'], 'No data')
+        self.assertFalse(response['success'])
+        self.assertIn('data', response['error'].lower())
+    
+    def test_process_request_none_input(self):
+        response = self.handler.process_request(None)
+        self.assertFalse(response['success'])"""
        
        scores = unified_diff_similarity_reward_func([patch], [test_patch], [generated_diff])
        
        # The score should be moderate because:
        # - Both files are modified (2/2 files match)
        # - The structure of changes is similar (validation, processing, error handling)
        # - Many lines are deleted/added in similar locations
        # But lower because:
        # - Different validation approach (ValueError vs TypeError/KeyError)
        # - Different processing steps (validator.validate vs sanitizer.clean)
        # - Different response format (status/message vs success/error)
        # - Different test assertions and method names
        # - The actual text content of most lines is quite different
        self.assertGreater(scores[0], 0.30, "Score should be moderate due to similar structure")
        self.assertLess(scores[0], 0.60, "Score should not be too high due to different specifics")


class TestUnifiedDiffFileMatchReward(unittest.TestCase):
    
    def test_file_match_perfect_match(self):
        """Test that all files matching gives score of 1.0."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old2
+    new2"""
        
        # Different content but same files
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -5,3 +5,3 @@
-    completely different
+    changes here

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -10,3 +10,3 @@
-    also different
+    content"""
        
        scores = unified_diff_file_match_reward_func([diff1], [diff2])
        self.assertEqual(scores[0], 1.0)
    
    def test_file_match_partial_match(self):
        """Test partial file matching."""
        # Patch touches 3 files
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old2
+    new2

diff --git a/file3.py b/file3.py
--- a/file3.py
+++ b/file3.py
@@ -1,3 +1,3 @@
-    old3
+    new3"""
        
        # Generated touches 2 out of 3 files
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new

diff --git a/file3.py b/file3.py
--- a/file3.py
+++ b/file3.py
@@ -1,3 +1,3 @@
-    old3
+    new3"""
        
        scores = unified_diff_file_match_reward_func([diff1], [diff2])
        self.assertAlmostEqual(scores[0], 2/3, places=2)
    
    def test_file_match_no_match(self):
        """Test completely different files."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        diff2 = """diff --git a/completely_different.py b/completely_different.py
--- a/completely_different.py
+++ b/completely_different.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        scores = unified_diff_file_match_reward_func([diff1], [diff2])
        self.assertEqual(scores[0], 0.0)
    
    def test_file_match_empty_generated(self):
        """Test when generated diff is empty."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        scores = unified_diff_file_match_reward_func([diff1], [""])
        self.assertEqual(scores[0], 0.0)
    
    def test_file_match_empty_patch(self):
        """Test when patch is empty but generated is not."""
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        scores = unified_diff_file_match_reward_func([""], [diff2])
        self.assertEqual(scores[0], 0.0)
    
    def test_file_match_both_empty(self):
        """Test when both patch and generated are empty."""
        scores = unified_diff_file_match_reward_func([""], [""])
        self.assertEqual(scores[0], 1.0)  # Both empty = perfect match
    
    def test_file_match_single_file(self):
        """Test single file matching."""
        diff1 = """diff --git a/single.py b/single.py
--- a/single.py
+++ b/single.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        # Same file
        diff2 = """diff --git a/single.py b/single.py
--- a/single.py
+++ b/single.py
@@ -10,3 +10,3 @@
-    different content
+    but same file"""
        
        scores = unified_diff_file_match_reward_func([diff1], [diff2])
        self.assertEqual(scores[0], 1.0)
    
    def test_file_match_batch_processing(self):
        """Test batch processing of multiple diffs."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        diff2 = """diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        # First pair: same file
        # Second pair: different files
        scores = unified_diff_file_match_reward_func([diff1, diff1], [diff1, diff2])
        self.assertEqual(len(scores), 2)
        self.assertEqual(scores[0], 1.0)
        self.assertEqual(scores[1], 0.0)
    
    def test_file_match_superset_files(self):
        """Test when generated diff has more files than patch."""
        # Patch touches 2 files
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old2
+    new2"""
        
        # Generated touches the same 2 files plus an extra one
        diff2 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old
+    new

diff --git a/file2.py b/file2.py
--- a/file2.py
+++ b/file2.py
@@ -1,3 +1,3 @@
-    old2
+    new2

diff --git a/extra_file.py b/extra_file.py
--- a/extra_file.py
+++ b/extra_file.py
@@ -1,3 +1,3 @@
-    extra
+    changes"""
        
        scores = unified_diff_file_match_reward_func([diff1], [diff2])
        # Should be 1.0 since all patch files are found
        self.assertEqual(scores[0], 1.0)
    
    def test_file_match_different_order(self):
        """Test that file order doesn't matter."""
        diff1 = """diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-    old
+    new

diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        # Same files but reversed order
        diff2 = """diff --git a/b.py b/b.py
--- a/b.py
+++ b/b.py
@@ -1,3 +1,3 @@
-    old
+    new

diff --git a/a.py b/a.py
--- a/a.py
+++ b/a.py
@@ -1,3 +1,3 @@
-    old
+    new"""
        
        scores = unified_diff_file_match_reward_func([diff1], [diff2])
        self.assertEqual(scores[0], 1.0)


if __name__ == "__main__":
    unittest.main()