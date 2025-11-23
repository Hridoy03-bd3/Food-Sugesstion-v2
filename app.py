import streamlit as st
import pandas as pd
import joblib
from datetime import datetime

# ------------------------------
# CSS DESIGN
# ------------------------------
st.markdown("""
<style>
body { background-color: #f0f2f6; }
h1 { color: #ff4b4b; text-align: center; }
.menu-card {
    background: #ffffff;
    padding: 15px;
    margin: 10px 0;
    border-radius: 12px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.12);
}
</style>
""", unsafe_allow_html=True)

# ------------------------------
# LOAD MODEL & ENCODERS
# ------------------------------
model = joblib.load("random_forest_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

st.title("🍽 Smart Meal Suggestion System For DIU Students")

# ------------------------------
# FOOD NUTRITION DATABASE
# ------------------------------
nutrition = {
    "Bhaat (Less) + Murgi (Chicken) Soup/Jhol + Shakh (Greens)": {"Calories": 520, "Protein": 32, "Carbs": 65, "Fat": 12},
    "Bhaat + Dim (Egg Bhuna) + Alu Bhorta + Daal": {"Calories": 610, "Protein": 28, "Carbs": 82, "Fat": 18},
    "Bhaat + Mach (Fish Curry) + Shakh (Leafy Greens) + Daal": {"Calories": 580, "Protein": 35, "Carbs": 70, "Fat": 14},
    "Khichuri (Light) + Dim Bhaji (Omelet)": {"Calories": 450, "Protein": 20, "Carbs": 60, "Fat": 10},
    "Khichuri (Light) + Dim Bhaji (Omelet) + Achaar": {"Calories": 490, "Protein": 21, "Carbs": 63, "Fat": 11},
    "Khichuri (Moderate/Heavy) + Murgi (Chicken) Curry + Shobji": {"Calories": 720, "Protein": 40, "Carbs": 88, "Fat": 20},
    "Ruti (3 pcs) + Dim (Egg) Curry + Daal": {"Calories": 520, "Protein": 22, "Carbs": 72, "Fat": 14},
    "Ruti/Porota (2 pcs) + Alu Bhaji (Potato) + Daal": {"Calories": 480, "Protein": 12, "Carbs": 68, "Fat": 16},
    "Ruti/Porota (2 pcs) + Shobji (Vegetable Curry) + Daal": {"Calories": 450, "Protein": 14, "Carbs": 62, "Fat": 12},
}

# ------------------------------
# CATEGORY OPTIONS
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
# USER INPUT SECTION
# ------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    Age = st.selectbox("Age ", Age_opt)
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
# SIMPLE ENCODER
# ------------------------------
def encode(col, val):
    le = label_encoders[col]
    return le.transform([val])[0]

input_data = pd.DataFrame({
    "Age ": [encode("Age ", Age)],
    "Gender": [encode("Gender", Gender)],
    "Height (in feet and inch)": [encode("Height (in feet and inch)", Height)],
    "Weight (kg)": [encode("Weight (kg)", Weight)],
    "Smoking Status": [encode("Smoking Status", Smoking)],
    "Are you a residential student (living in a hall/hostel)?": [encode("Are you a residential student (living in a hall/hostel)?", Resident)],
    "Marital Status": [encode("Marital Status", Marital)],
    "How much sleep did you get last night?": [encode("How much sleep did you get last night?", Sleep)],
    "How would you describe your current stress/anxiety level?": [encode("How would you describe your current stress/anxiety level?", Stress)],
    "What is your estimated physical activity level for today?": [encode("What is your estimated physical activity level for today?", Activity)],
    "How long ago was your last proper meal (e.g., breakfast)?": [encode("How long ago was your last proper meal (e.g., breakfast)?", LastMeal)],
    "How would you rate your current feeling of hunger?": [encode("How would you rate your current feeling of hunger?", Hunger)],
    "Have you already skipped a meal today (Breakfast/Lunch/Dinner)?": [encode("Have you already skipped a meal today (Breakfast/Lunch/Dinner)?", Skipped)],
    "What meal are you likely to take next?": [encode("What meal are you likely to take next?", NextMeal)],
})

# ------------------------------
# PREDICTION BUTTON
# ------------------------------
if st.button("🔍 Get Meal Suggestion"):
    pred = model.predict(input_data)[0]
    meal = label_encoders["Meal Suggestion"].inverse_transform([pred])[0]

    st.success(f"🍽 **Recommended Meal:** {meal}")
    
    # Show nutrition
    if meal in nutrition:
        nut = nutrition[meal]
        st.subheader("🍎 Nutrition Breakdown")
        st.write(f"**Calories:** {nut['Calories']} kcal")
        st.write(f"**Protein:** {nut['Protein']} g")
        st.write(f"**Carbohydrates:** {nut['Carbs']} g")
        st.write(f"**Fat:** {nut['Fat']} g")

# ------------------------------
# FOOD MENU VIEWER
# ------------------------------
st.header("📋 Full Food Menu with Nutrition")

meal_type = st.radio("Select meal:", ["Breakfast", "Lunch", "Dinner"])

if meal_type == "Breakfast":
    items = ["Khichuri (Light) + Dim Bhaji (Omelet)",
             "Ruti/Porota (2 pcs) + Alu Bhaji (Potato) + Daal",
             "Ruti/Porota (2 pcs) + Shobji (Vegetable Curry) + Daal"]
elif meal_type == "Lunch":
    items = ["Bhaat (Less) + Murgi (Chicken) Soup/Jhol + Shakh (Greens)",
             "Bhaat + Mach (Fish Curry) + Shakh (Leafy Greens) + Daal",
             "Khichuri (Moderate/Heavy) + Murgi (Chicken) Curry + Shobji"]
else:
    items = ["Bhaat + Dim (Egg Bhuna) + Alu Bhorta + Daal",
             "Ruti (3 pcs) + Dim (Egg) Curry + Daal"]

# show menu
for item in items:
    st.markdown(f"<div class='menu-card'><b>{item}</b><br>"
                f"Calories: {nutrition[item]['Calories']} kcal<br>"
                f"Protein: {nutrition[item]['Protein']} g<br>"
                f"Carbs: {nutrition[item]['Carbs']} g<br>"
                f"Fat: {nutrition[item]['Fat']} g</div>",
                unsafe_allow_html=True)
