import unittest
from src.rewards.diff import split_diff_by_files, normalize_file_diff, unified_diff_similarity_reward_func


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
    
    def test_normalize_file_diff(self):
        """Test extracting filename and changes from a file diff."""
        file_diff = """diff --git a/src/example.py b/src/example.py
--- a/src/example.py
+++ b/src/example.py
@@ -1,3 +1,3 @@
 def hello():
-    print("hello")
+    print("hello world")
 return None"""
        
        filename, changes = normalize_file_diff(file_diff)
        self.assertEqual(filename, "src/example.py")
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff1])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
        self.assertEqual(scores[0], 0.0)
    
    def test_unified_diff_similarity_empty_generated(self):
        """Test when generated diff is empty string."""
        diff1 = """diff --git a/file1.py b/file1.py
--- a/file1.py
+++ b/file1.py
@@ -1,3 +1,3 @@
-    old line
+    new line"""
        
        scores = unified_diff_similarity_reward_func([diff1], [""])
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
        scores = unified_diff_similarity_reward_func([diff1, diff1], [diff1, diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
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
        
        scores = unified_diff_similarity_reward_func([diff1], [diff2])
        # Should be perfect since we only compare +/- lines
        self.assertEqual(scores[0], 1.0)


if __name__ == "__main__":
    unittest.main()