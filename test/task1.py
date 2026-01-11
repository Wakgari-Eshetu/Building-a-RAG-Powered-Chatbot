import unittest
import pandas as pd
import os

DATA_DIR = os.path.join("..", "data", "processed")
FILTERED_FILE = os.path.join(DATA_DIR, "filtered_complaints.csv")

class TestTask1(unittest.TestCase):

    def setUp(self):
        self.df = pd.read_csv(FILTERED_FILE)

    def test_non_empty(self):
        
        self.assertFalse(self.df.empty, "Filtered dataset is empty!")

    def test_required_columns(self):
        
        expected_columns = [
            'Date received', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
            'Consumer complaint narrative', 'Company public response', 'Company',
            'State', 'ZIP code', 'Tags', 'Consumer consent provided?',
            'Submitted via', 'Date sent to company', 'Company response to consumer',
            'Timely response?', 'Consumer disputed?', 'Complaint ID'
        ]
        for col in expected_columns:
            self.assertIn(col, self.df.columns, f"Missing column: {col}")

    def test_no_empty_narratives(self):
        
        self.assertFalse(self.df['Consumer complaint narrative'].isna().any(),
                         "There are empty complaint narratives!")

    def test_products_filtered(self):
        
        allowed_products = ['Credit card', 'Personal loan', 'Savings account', 'Money transfers']
        products = self.df['Product'].unique()
        for p in products:
            self.assertIn(p, allowed_products, f"Unexpected product: {p}")

if __name__ == "__main__":
    unittest.main()
