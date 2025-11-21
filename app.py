import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------------------
# CSS STYLING
# ------------------------------
st.markdown("""
<style>
body { background-color: #f0f2f6; }
h1 { color: #ff4b4b; text-align: center; font-family: 'Arial', sans-serif; margin-bottom: 10px; }
h3 { color: #333333; font-family: 'Arial', sans-serif; }

/* Input cards */
.stSelectbox, .stButton {
    background-color: #ffffff;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.12);
    margin-bottom: 12px;
}

/* Button styling */
.stButton>button {
    background: linear-gradient(90deg, #ff4b4b, #ff1a1a);
    color: white;
    font-weight: bold;
    border-radius: 12px;
    padding: 12px 25px;
    border: none;
    cursor: pointer;
    transition: 0.3s ease all;
}
.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #ff1a1a, #ff4b4b);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# Load model & encoders
# ------------------------------
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🍽 Smart Meal Suggestion System")
st.write("Fill the fields below to get your personalized meal recommendation.")

# ------------------------------
# CATEGORY OPTIONS
# ------------------------------
Age_opt = ["18-20", "21-23", "23-26"]
Gender_opt = ["Female", "Gender", "Male"]
Height_opt = [
    "5 feet 0 inches – 5 feet 3 inches",
    "5 feet 4 inches – 5 feet 7 inches",
    "5 feet 8 inches – 5 feet 11 inches",
    "6 feet 0 inches or Above",
    "Below 5 feet 0 inches"
]
Weight_opt = [
    "45 kg – 55 kg",
    "56 kg – 65 kg",
    "66 kg – 75 kg",
    "76 kg or Above (Potentially Overweight/Obese)",
    "Below 45 kg (Potentially Underweight)"
]
Smoke_opt = ["No", "Yes"]
Resident_opt = ["No", "Yes"]
Marital_opt = ["Married", "Unmarried"]
Sleep_opt = [
    "Excessive (9+ hours)",
    "Low Sleep (Less than 5 hours)",
    "Moderate (5-6 hours)",
    "Optimal (7-8 hours)"
]
Stress_opt = ["High/Exam Stress", "Low/Calm", "Moderate"]
Activity_opt = [
    "Light (Walking to class/stairs)",
    "Moderate (Gym/Sports less than 60 mins)",
    "Sedentary (Mostly sitting/studying )",
    "Vigorous (Sports more than 60 mins/Heavy labor)"
]
LastMeal_opt = [
    "2-4 hours ago",
    "4-6 hours ago (Optimal hunger)",
    "Less than 2 hours ago",
    "More than 6 hours ago (Skipped meal/High hunger)"
]
Hunger_opt = [
    "Moderately hungry (Ready to eat)",
    "Not hungry at all",
    "Slightly hungry",
    "Very hungry (Feeling weak/distracted)"
]
Skip_opt = [
    "No, I have not skipped a meal.",
    "Yes, I skipped Breakfast",
    "Yes, I skipped Dinner",
    "Yes, I skipped Lunch",
]
NextMeal_opt = ["Breakfast", "Dinner", "Lunch"]

# ------------------------------
# STREAMLIT INPUTS IN COLUMNS
# ------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.selectbox("Age ", Age_opt, index=0)
    Gender = st.selectbox("Gender", Gender_opt, index=0)
    Height = st.selectbox("Height (in feet and inch)", Height_opt, index=0)
    Weight = st.selectbox("Weight (kg)", Weight_opt, index=0)

with col2:
    Smoking = st.selectbox("Smoking Status", Smoke_opt, index=0)
    Resident = st.selectbox("Residential Student?", Resident_opt, index=0)
    Marital = st.selectbox("Marital Status", Marital_opt, index=0)
    Sleep = st.selectbox("Sleep Last Night", Sleep_opt, index=0)

with col3:
    Stress = st.selectbox("Stress/Anxiety Level", Stress_opt, index=0)
    Activity = st.selectbox("Physical Activity Level", Activity_opt, index=0)
    LastMeal = st.selectbox("Last Meal Time", LastMeal_opt, index=0)
    Hunger = st.selectbox("Current Hunger Level", Hunger_opt, index=0)
    Skipped = st.selectbox("Skipped a Meal Today?", Skip_opt, index=0)
    NextMeal = st.selectbox("Next Meal?", NextMeal_opt, index=0)

# ------------------------------
# ENCODE FUNCTION
# ------------------------------
def encode_value(col, value):
    le = label_encoders[col]
    value = value.strip()
    if col == "Have you already skipped a meal today (Breakfast/Lunch/Dinner)?":
        value = value.rstrip(',')
    if col == "What meal are you likely to take next?":
        if value.lower() == "breakfast":
            value = "Breakfasst"
    if value not in le.classes_:
        st.error(f"Value '{value}' not found in encoder for '{col}'")
        return None
    return le.transform([value])[0]

# ------------------------------
# CREATE DATAFRAME
# ------------------------------
input_data = pd.DataFrame({
    "Age ": [encode_value("Age ", Age)],
    "Gender": [encode_value("Gender", Gender)],
    "Height (in feet and inch)": [encode_value("Height (in feet and inch)", Height)],
    "Weight (kg)": [encode_value("Weight (kg)", Weight)],
    "Smoking Status": [encode_value("Smoking Status", Smoking)],
    "Are you a residential student (living in a hall/hostel)?": [encode_value("Are you a residential student (living in a hall/hostel)?", Resident)],
    "Marital Status": [encode_value("Marital Status", Marital)],
    "How much sleep did you get last night?": [encode_value("How much sleep did you get last night?", Sleep)],
    "How would you describe your current stress/anxiety level?": [encode_value("How would you describe your current stress/anxiety level?", Stress)],
    "What is your estimated physical activity level for today?": [encode_value("What is your estimated physical activity level for today?", Activity)],
    "How long ago was your last proper meal (e.g., breakfast)?": [encode_value("How long ago was your last proper meal (e.g., breakfast)?", LastMeal)],
    "How would you rate your current feeling of hunger?": [encode_value("How would you rate your current feeling of hunger?", Hunger)],
    "Have you already skipped a meal today (Breakfast/Lunch/Dinner)?": [encode_value("Have you already skipped a meal today (Breakfast/Lunch/Dinner)?", Skipped)],
    "What meal are you likely to take next?": [encode_value("What meal are you likely to take next?", NextMeal)],
})

# ------------------------------
# PREDICT AND SAVE
# ------------------------------
if st.button("🔍 Get Meal Suggestion"):
    if input_data.isnull().values.any():
        st.error("Some inputs could not be encoded. Please check your selections.")
    else:
        # Predict
        encoded_pred = model.predict(input_data)[0]
        meal_decoder = label_encoders["Meal Suggestion"]
        final_meal = meal_decoder.inverse_transform([encoded_pred])[0]

        st.success("🍽 **Recommended Meal:**")
        st.subheader(final_meal)

        # Save to CSV
        record = input_data.copy()
        record["Meal Suggestion"] = final_meal
        record["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            history_file = "meal_history.csv"
            df_history = pd.read_csv(history_file)
            df_history = pd.concat([df_history, record], ignore_index=True)
        except FileNotFoundError:
            df_history = record
        df_history.to_csv(history_file, index=False)

        st.info(f"✅ Your suggestion has been saved to {history_file}")
