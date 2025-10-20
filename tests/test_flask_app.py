import unittest
import importlib

class StreamlitAppTests(unittest.TestCase):
    def test_import_app(self):
        """
        Test that the Streamlit app can be imported successfully
        without runtime errors (e.g., missing dependencies, syntax issues).
        """
        try:
            module = importlib.import_module("asthama_app.app")
        except Exception as e:
            self.fail(f"Failed to import Streamlit app: {e}")

    def test_prediction_function(self):
        """
        Test your model prediction function directly
        (not through Streamlit UI).
        """
        from src.model.predict import predict_asthma  # adjust this path to your actual prediction function

        sample_input = {
            "Age": 45,
            "BMI": 23.4,
            "Family_History": 1,
            "Air_Pollution_Level": "Moderate",
            "Physical_Activity_Level": "Active",
            "Occupation_Type": "Indoor",
            "Allergies": "Dust",
            "Comorbidities": "None",
            "Medication_Adherence": 1,
            "Number_of_ER_Visits": 0,
            "Peak_Expiratory_Flow": 350.5,
            "FeNO_Level": 15.2,
            "Gender": "Male",
            "Smoking_Status": "Never"
        }

        try:
            result = predict_asthma(sample_input)
        except Exception as e:
            self.fail(f"Prediction function crashed: {e}")

        # Example: assert that output is a string or label
        self.assertIn(result, ["Asthma", "No Asthma"], "Prediction output is invalid")

if __name__ == "__main__":
    unittest.main()
