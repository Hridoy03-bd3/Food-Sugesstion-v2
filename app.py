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
h3 { color: #333333; font-family: 'Arial', sans-serif'; }

.stSelectbox, .stButton {
    background-color: #ffffff;
    padding: 12px;
    border-radius: 12px;
    box-shadow: 2px 2px 12px rgba(0,0,0,0.12);
    margin-bottom: 12px;
}

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
# LOAD MODEL & ENCODERS
# ------------------------------
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🍽 Smart Meal Suggestion System For DIU Students")
st.write("Fill the fields below to get your personalized meal recommendation.")

# ------------------------------
# OPTIONS
# ------------------------------
Age_opt = ["18-20", "21-23", "23-26"]
Gender_opt = ["Female", "Male"]
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
Sleep_opt = ["Excessive (9+ hours)", "Low Sleep (Less than 5 hours)", "Moderate (5-6 hours)", "Optimal (7-8 hours)"]
Stress_opt = ["High/Exam Stress", "Low/Calm", "Moderate"]
Activity_opt = ["Light (Walking to class/stairs)", "Moderate (Gym/Sports <60 mins)", "Sedentary (Mostly sitting/studying)", "Vigorous (Sports >60 mins/Heavy labor)"]
LastMeal_opt = ["2-4 hours ago", "4-6 hours ago (Optimal hunger)", "Less than 2 hours ago", "More than 6 hours ago (Skipped meal/High hunger)"]
Hunger_opt = ["Moderately hungry (Ready to eat)", "Not hungry at all", "Slightly hungry", "Very hungry (Feeling weak/distracted)"]
Skip_opt = ["No, I have not skipped a meal.", "Yes, I skipped Breakfast", "Yes, I skipped Dinner", "Yes, I skipped Lunch"]
NextMeal_opt = ["", "Breakfast", "Lunch", "Dinner"]

# ------------------------------
# USER INPUT
# ------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    Age = st.selectbox("Age", Age_opt)
    Gender = st.selectbox("Gender", Gender_opt)
    Height = st.selectbox("Height (in feet and inch)", Height_opt)
    Weight = st.selectbox("Weight (kg)", Weight_opt)

with col2:
    Smoking = st.selectbox("Smoking Status", Smoke_opt)
    Resident = st.selectbox("Residential (Inside Campus)?", Resident_opt)
    Marital = st.selectbox("Marital Status", Marital_opt)
    Sleep = st.selectbox("Sleep Last Night", Sleep_opt)

with col3:
    Stress = st.selectbox("Stress/Anxiety Level", Stress_opt)
    Activity = st.selectbox("Physical Activity Level", Activity_opt)
    LastMeal = st.selectbox("Last Meal Time", LastMeal_opt)
    Hunger = st.selectbox("Current Hunger Level", Hunger_opt)
    Skipped = st.selectbox("Skipped a Meal Today?", Skip_opt)
    NextMeal = st.selectbox("Next Meal?", NextMeal_opt)

# ------------------------------
# BMI CALCULATION
# ------------------------------
def calculate_bmi(height, weight):
    height_ft, height_inch = height.split(" ")[0], height.split(" ")[2]  # Example: "5 feet 8 inches"
    height_ft = int(height_ft)
    height_inch = int(height_inch)
    
    height_in_meters = (height_ft * 0.3048) + (height_inch * 0.0254)
    
    bmi = weight / (height_in_meters ** 2)
    
    return round(bmi, 2)

# ------------------------------
# CALORIES BASED ON BMI
# ------------------------------
def calculate_calories(bmi):
    if bmi < 18.5:
        calorie_needs = "Higher than normal calorie intake to gain weight (2500-3000 kcal/day)"
    elif 18.5 <= bmi <= 24.9:
        calorie_needs = "Normal calorie intake (2000-2500 kcal/day)"
    elif 25 <= bmi <= 29.9:
        calorie_needs = "Moderate calorie intake (1800-2200 kcal/day) for weight management"
    else:
        calorie_needs = "Lower calorie intake (1500-2000 kcal/day) for weight loss"
    
    return calorie_needs

# ------------------------------
# SAFE ENCODE FUNCTION
# ------------------------------
def safe_encode(col, val):
    le = label_encoders[col]
    val = val.strip()
    if col == "What meal are you likely to take next?":
        if val == "Breakfast" and "Breakfasst" in le.classes_:
            val = "Breakfasst"
    if val not in le.classes_:
        st.warning(f"Value '{val}' not found in encoder for '{col}', using default.")
        val = le.classes_[0]
    return le.transform([val])[0]

# ------------------------------
# PREPARE DATAFRAME
# ------------------------------
input_df = pd.DataFrame({
    "Age ": [safe_encode("Age ", Age)],
    "Gender": [safe_encode("Gender", Gender)],
    "Height (in feet and inch)": [safe_encode("Height (in feet and inch)", Height)],
    "Weight (kg)": [safe_encode("Weight (kg)", Weight)],
    "Smoking Status": [safe_encode("Smoking Status", Smoking)],
    "Are you a residential student (living in a hall/hostel)?": [safe_encode("Are you a residential student (living in a hall/hostel)?", Resident)],
    "Marital Status": [safe_encode("Marital Status", Marital)],
    "How much sleep did you get last night?": [safe_encode("How much sleep did you get last night?", Sleep)],
    "How would you describe your current stress/anxiety level?": [safe_encode("How would you describe your current stress/anxiety level?", Stress)],
    "What is your estimated physical activity level for today?": [safe_encode("What is your estimated physical activity level for today?", Activity)],
    "How long ago was your last proper meal (e.g., breakfast)?": [safe_encode("How long ago was your last proper meal (e.g., breakfast)?", LastMeal)],
    "How would you rate your current feeling of hunger?": [safe_encode("How would you rate your current feeling of hunger?", Hunger)],
    "Have you already skipped a meal today (Breakfast/Lunch/Dinner)?": [safe_encode("Have you already skipped a meal today (Breakfast/Lunch/Dinner)?", Skipped)],
    "What meal are you likely to take next?": [safe_encode("What meal are you likely to take next?", NextMeal)],
})

# ------------------------------
# PREDICTION
# ------------------------------
if st.button("🔍 Get Meal Suggestion"):
    # Get weight (kg) value from the user
    weight_value = int(Weight.split(" ")[0])  # Example: "45 kg – 55 kg" -> 45
    
    # Calculate BMI and calorie needs
    bmi_value = calculate_bmi(Height, weight_value)
    calorie_suggestion = calculate_calories(bmi_value)

    # Show BMI and calorie suggestion
    st.subheader(f"BMI: {bmi_value}")
    st.write(f"Calories Needed: {calorie_suggestion}")
    
    # Meal prediction
    pred_encoded = model.predict(input_df)[0]
    meal_decoder = label_encoders["Meal Suggestion"]
    final_meal = meal_decoder.inverse_transform([pred_encoded])[0
