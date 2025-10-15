import unittest
from asthama_app.app import app

class FlaskAppTests(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_home_page(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Asthma Prediction</title>', response.data)

    def test_predict_page(self):
        # Provide sample valid data matching your form
        response = self.app.post('/predict', data={
            'Age': 45,
            'BMI': 23.4,
            'Family_History': 1,
            'Air_Pollution_Level': 'Moderate',
            'Physical_Activity_Level': 'Active',
            'Occupation_Type': 'Indoor',
            'Allergies': 'Dust',
            'Comorbidities': 'None',
            'Medication_Adherence': 1,
            'Number_of_ER_Visits': 0,
            'Peak_Expiratory_Flow': 350.5,
            'FeNO_Level': 15.2,
            'Gender': 'Male',
            'Smoking_Status': 'Never'
        })
        self.assertEqual(response.status_code, 200)
        # Adjust to what your app actually returns
        self.assertTrue(
            b'Asthma' in response.data or b'No Asthma' in response.data,
            "Response should contain either 'Asthma' or 'No Asthma'"
        )
