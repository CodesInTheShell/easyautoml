import unittest
import helpers 
  
class HelperTest(unittest.TestCase): 
  
	def test_create_project_dir(self):

		print(helpers.create_project_dir('testdir'))
		# self.assertTrue(helpers.create_project_dir('testdir'))
  
if __name__ == '__main__':
	unittest.main() 
