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
NextMeal_opt = ["", "Breakfast", "Lunch", "Dinner"]  # empty string default hides menu

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
# SAFE ENCODE FUNCTION
# ------------------------------
def safe_encode(col, val):
    le = label_encoders[col]
    val = val.strip()
    # Map values to match training labels if necessary
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
    pred_encoded = model.predict(input_df)[0]
    meal_decoder = label_encoders["Meal Suggestion"]
    final_meal = meal_decoder.inverse_transform([pred_encoded])[0]

    st.success("🍽 Recommended Meal:")
    st.subheader(final_meal)

    # Save in session_state
    st.session_state["last_prediction"] = final_meal
    st.session_state["last_input"] = input_df.copy()

# ------------------------------
# SAVE TO CSV
# ------------------------------
if "last_prediction" in st.session_state:
    if st.checkbox("💾 Save this suggestion to history?"):
        record = st.session_state["last_input"]
        record["Meal Suggestion"] = st.session_state["last_prediction"]
        record["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        try:
            df_hist = pd.read_csv("meal_history.csv")
            df_hist = pd.concat([df_hist, record], ignore_index=True)
        except FileNotFoundError:
            df_hist = record
        df_hist.to_csv("meal_history.csv", index=False)
        st.info("Saved successfully.")

# ------------------------------
# FOOD MENU BASED ON NEXT MEAL
# ------------------------------
nutrition_data = {
    "Bhaat (Less) + Murgi (Chicken) Soup/Jhol + Shakh (Greens)": {"Calories": 400, "Protein": 30, "Carbs": 50, "Fat": 12},
    "Bhaat + Dim (Egg Bhuna) + Alu Bhorta + Daal": {"Calories": 520, "Protein": 25, "Carbs": 65, "Fat": 18},
    "Bhaat + Mach (Fish Curry) + Shakh (Leafy Greens) + Daal": {"Calories": 500, "Protein": 35, "Carbs": 60, "Fat": 15},
    "Khichuri (Light) + Dim Bhaji (Omelet)": {"Calories": 450, "Protein": 20, "Carbs": 55, "Fat": 12},
    "Khichuri (Light) + Dim Bhaji (Omelet) + Achaar": {"Calories": 470, "Protein": 22, "Carbs": 57, "Fat": 13},
    "Khichuri (Moderate/Heavy) + Murgi (Chicken) Curry + Shobji": {"Calories": 550, "Protein": 30, "Carbs": 65, "Fat": 20},
    "Ruti (3 pcs) + Dim (Egg) Curry + Daal": {"Calories": 480, "Protein": 25, "Carbs": 55, "Fat": 15},
    "Ruti/Porota (2 pcs) + Alu Bhaji (Potato) + Daal": {"Calories": 400, "Protein": 15, "Carbs": 60, "Fat": 10},
    "Ruti/Porota (2 pcs) + Shobji (Vegetable Curry) + Daal": {"Calories": 420, "Protein": 18, "Carbs": 62, "Fat": 12},
    "Ruti/Porota (2 pcs) + Shobji + Daal": {"Calories": 410, "Protein": 17, "Carbs": 60, "Fat": 11},
}

# Show menu only if user selects a NextMeal
if NextMeal != "":
    st.header(f"📋 {NextMeal} Menu")
    if NextMeal == "Breakfast":
        menu_items = ["Khichuri (Light) + Dim Bhaji (Omelet)", "Khichuri (Light) + Dim Bhaji (Omelet) + Achaar",
                      "Ruti/Porota (2 pcs) + Alu Bhaji (Potato) + Daal", "Ruti/Porota (2 pcs) + Shobji + Daal"]
    elif NextMeal == "Lunch":
        menu_items = ["Bhaat (Less) + Murgi (Chicken) Soup/Jhol + Shakh (Greens)",
                      "Bhaat + Mach (Fish Curry) + Shakh (Leafy Greens) + Daal",
                      "Khichuri (Moderate/Heavy) + Murgi (Chicken) Curry + Shobji",
                      "Bhaat + Dim (Egg Bhuna) + Alu Bhorta + Daal"]
    else:  # Dinner
        menu_items = ["Ruti (3 pcs) + Dim (Egg) Curry + Daal",
                      "Ruti/Porota (2 pcs) + Shobji (Vegetable Curry) + Daal","Khichuri (Light) + Dim Bhaji (Omelet)"]

    for item in menu_items:
        st.write(f"• **{item}**")
        if item in nutrition_data:
            nut = nutrition_data[item]
            st.write(f" Calories: {nut['Calories']} kcal | Protein: {nut['Protein']} g | Carbs: {nut['Carbs']} g | Fat: {nut['Fat']} g")


