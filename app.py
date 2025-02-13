import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from io import BytesIO
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
from reportlab.lib import colors
import heapq

st.title("️🧑‍🍳Personalized Meal Planner")

# ---- User Inputs ----
st.subheader("Personalize Your Meal Plan")
user_calorie_goal = st.number_input("Enter your daily calorie goal", min_value=1000, max_value=4000, step=100, value=2000)

# ---- User Inputs for Macronutrient Ratios ----
st.subheader("Adjust Macronutrient Ratios")

user_carbs_ratio = st.slider("Carbs (%)", 0, 100, 50)
user_protein_ratio = st.slider("Protein (%)", 0, 100, 30)
user_fats_ratio = st.slider("Fats (%)", 0, 100, 20)

if user_carbs_ratio + user_protein_ratio + user_fats_ratio != 100:
    st.error("The total percentage of carbs, protein, and fats must equal 100%. Please adjust the sliders.")

# ---- Diet Preference ----
diet_preference = st.selectbox("Diet Preference", ["No Preference", "Vegetarian", "Vegan", "Keto"])

# ---- Sample Food Database ----
food_data = [
    {"name": "Chicken Breast", "calories": 200, "protein": 30, "carbs": 0, "fats": 5},
    {"name": "Brown Rice", "calories": 215, "protein": 5, "carbs": 45, "fats": 2},
    {"name": "Broccoli", "calories": 55, "protein": 4, "carbs": 11, "fats": 0},
    {"name": "Avocado", "calories": 160, "protein": 2, "carbs": 8, "fats": 15},
    {"name": "Eggs", "calories": 70, "protein": 6, "carbs": 1, "fats": 5},
    {"name": "Almonds", "calories": 160, "protein": 6, "carbs": 6, "fats": 14},
    {"name": "Salmon", "calories": 180, "protein": 25, "carbs": 0, "fats": 10},
    {"name": "Quinoa", "calories": 220, "protein": 8, "carbs": 39, "fats": 3.5},
    {"name": "Sweet Potato", "calories": 112, "protein": 2, "carbs": 26, "fats": 0},
    {"name": "Spinach", "calories": 23, "protein": 2.9, "carbs": 3.6, "fats": 0.4},
    {"name": "Greek Yogurt", "calories": 100, "protein": 10, "carbs": 6, "fats": 0},
    {"name": "Tuna", "calories": 130, "protein": 28, "carbs": 0, "fats": 1},
    {"name": "Peanut Butter", "calories": 190, "protein": 8, "carbs": 6, "fats": 16},
    {"name": "Oats", "calories": 150, "protein": 5, "carbs": 27, "fats": 3},
    {"name": "Cottage Cheese", "calories": 90, "protein": 11, "carbs": 3, "fats": 4},
    {"name": "Banana", "calories": 105, "protein": 1.3, "carbs": 27, "fats": 0.3},
    {"name": "Carrots", "calories": 41, "protein": 0.9, "carbs": 9.6, "fats": 0.2},
    {"name": "Chia Seeds", "calories": 138, "protein": 5, "carbs": 12, "fats": 9},
    {"name": "Cheddar Cheese", "calories": 115, "protein": 7, "carbs": 1, "fats": 9},
    {"name": "Tomatoes", "calories": 22, "protein": 1.1, "carbs": 4.8, "fats": 0.2},
    {"name": "Lentils", "calories": 230, "protein": 18, "carbs": 40, "fats": 1},
    {"name": "Chickpeas", "calories": 164, "protein": 9, "carbs": 27, "fats": 2.6},
    {"name": "Zucchini", "calories": 33, "protein": 2.4, "carbs": 6.1, "fats": 0.6},
    {"name": "Tofu", "calories": 80, "protein": 8, "carbs": 2, "fats": 4.5},
    {"name": "Mushrooms", "calories": 22, "protein": 3.1, "carbs": 3.3, "fats": 0.3}
]

df_food = pd.DataFrame(food_data)


# ---- Clustering Foods ----
def cluster_foods(food):
    nutrients = food[["protein", "carbs", "fats"]]
    scaler = StandardScaler()
    nutrients_normalized = scaler.fit_transform(nutrients)

    kmeans = KMeans(n_clusters=3, random_state=42)
    food["Cluster"] = kmeans.fit_predict(nutrients_normalized)

    cluster_map = {0: "High Protein", 1: "High Carb", 2: "Balanced"}
    food["ClusterLabel"] = food["Cluster"].map(cluster_map)

    return food


# Apply clustering
df_food = cluster_foods(df_food)

# ---- Display Clustered Food Database ----
st.subheader("Clustered Food Database")
st.dataframe(df_food[["name", "ClusterLabel", "calories", "protein", "carbs", "fats"]])

# ---- Dietary Preference Filtering ----
st.sidebar.subheader("Dietary Preferences")
selected_cluster = st.sidebar.selectbox(
    "Select Food Cluster:",
    options=["All", "High Protein", "High Carb", "Balanced"]
)

if selected_cluster != "All":
    df_food = df_food[df_food["ClusterLabel"] == selected_cluster]

# ---- Meal Plan Generation ----
def generate_meal_plan_with_heap(food_df, calorie_goal, carbs_ratio, protein_ratio, fats_ratio):
    target_carbs_plan = (calorie_goal * carbs_ratio) / 400
    target_protein = (calorie_goal * protein_ratio) / 400
    target_fats_plan = (calorie_goal * fats_ratio) / 900

    food_df['priority'] = (
            (0.4 * food_df['calories'] / calorie_goal) +
            (0.3 * food_df['protein'] / target_protein) +
            (0.2 * food_df['carbs'] / target_carbs_plan) +
            (0.1 * food_df['fats'] / target_fats_plan)
    )

    # Create a heap based on priority (negating the priority to create a max-heap)
    heap = []
    for _, food in food_df.iterrows():
        heapq.heappush(heap, (-food['priority'], food))  # Use negative priority for max-heap

    # Generate meal plan
    meal_plan = []
    total_calories, total_carbs, total_protein, total_fats = 0, 0, 0, 0

    while heap and (total_calories < calorie_goal or total_carbs < target_carbs_plan or
                    total_protein < target_protein or total_fats < target_fats_plan):
        _, food = heapq.heappop(heap)
        meal_plan.append(food)
        total_calories += food['calories']
        total_carbs += food['carbs']
        total_protein += food['protein']
        total_fats += food['fats']

    return pd.DataFrame(meal_plan), total_calories, total_carbs, total_protein, total_fats

meal_plan_result, total_calories_plan, total_carbs_plan, total_protein_plan, total_fats_plan = \
    generate_meal_plan_with_heap(df_food, user_calorie_goal, user_carbs_ratio / 100, user_protein_ratio / 100, user_fats_ratio / 100)

# ---- Display Meal Plan ----
st.subheader("Your Personalized Meal Plan")
st.dataframe(meal_plan_result[['name', 'calories', 'protein', 'carbs', 'fats']])

# ---- Macronutrient Breakdown Pie Chart ----
macros = {
    "Carbs": total_carbs_plan,
    "Protein": total_protein_plan,
    "Fats": total_fats_plan,
}
fig = px.pie(names=macros.keys(), values=macros.values(), title="Macronutrient Breakdown")
st.plotly_chart(fig)

# ---- Meal-by-Meal Stacked Bar Chart ----
meal_plan_result['Meal'] = [f"Meal {i + 1}" for i in range(len(meal_plan_result))]
fig = px.bar(
    meal_plan_result, x="Meal", y=["protein", "carbs", "fats"],
    title="Macronutrient Contributions per Meal",
    labels={"value": "Grams", "variable": "Macronutrient"},
    barmode="stack",
    color_discrete_map={"protein": "blue", "carbs": "orange", "fats": "purple"}
)

st.plotly_chart(fig)

def generate_pdf(meal_plan):
    buffer = BytesIO()  # Create an in-memory buffer

    # Create the PDF using the buffer
    pdf = SimpleDocTemplate(buffer)

    # Prepare the data for the table in the PDF
    data = [["Name", "Calories", "Protein (g)", "Carbs (g)", "Fats (g)"]]
    for _, row in meal_plan.iterrows():
        data.append([row["name"], row["calories"], row["protein"], row["carbs"], row["fats"]])

    # Create the table and apply style
    table = Table(data)
    style = TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
    ])
    table.setStyle(style)

    # Build the PDF document (it will be stored in the buffer)
    pdf.build([table])

    # Go to the beginning of the buffer to prepare for download
    buffer.seek(0)

    return buffer

# ---- Provide a download link for the PDF ----
if not meal_plan_result.empty:
    st.subheader("Export Your Meal Plan")
    pdf_buffer = generate_pdf(meal_plan_result)  # Generate PDF in memory
    st.download_button(
        label="Download Meal Plan PDF",
        data=pdf_buffer,
        file_name="meal_plan.pdf",
        mime="application/pdf"
    )
