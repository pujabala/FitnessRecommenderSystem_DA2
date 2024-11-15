import streamlit as st
import pandas as pd
import numpy as np
import pickle
from RecommendationSystem import FinalRecommenderSystem  # Import the function

def user_input():
    st.title("Fitness Recommendation System")
    age = st.number_input("Age", min_value=18, max_value=100, value=29)
    gender = st.selectbox("Gender", ["Male", "Female"])
    height = st.number_input("Height (cm)", min_value=100, max_value=250, value=164)
    weight = st.number_input("Weight (kg)", min_value=30, max_value=200, value=91)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=33.83)
    primary_fitness_goal = st.selectbox("Primary Fitness Goal", 
                                        ["Endurance", "Muscle gain", "Weight loss", "General fitness"])
    secondary_fitness_goal = st.selectbox("Secondary Fitness Goal", 
                                          ["Stress reduction", "Flexibility", "Balance", "Posture correction"])
    preferred_activity_type = st.selectbox("Preferred Activity Type", 
                                          ["HIIT", "Cardio", "Yoga", "Swimming", "Walking"])
    workout_intensity = st.selectbox("Workout Intensity", ["Beginner", "Intermediate", "Expert"])

    given_user_df = pd.DataFrame([{
        'uid': 1012,
        'Age': age,
        'Gender': gender,
        'Height (cm)': height,
        'Weight (kg)': weight,
        'BMI': bmi,
        'Primary Fitness Goal': primary_fitness_goal,
        'Secondary Fitness Goal': secondary_fitness_goal,
        'Preferred Activity Type': preferred_activity_type,
        'Workout Intensity': workout_intensity
    }])
    print(given_user_df)
    return given_user_df

def transform_schedule(df):
    df['Exercises'] = df['Exercises'].str.split(',')
    df_exploded = df.explode('Exercises').reset_index(drop=True)
    df_exploded['exercise_index'] = df_exploded.groupby(['Day']).cumcount()
    df_pivoted = df_exploded.pivot(index='exercise_index', columns='Day', values='Exercises')
    df_pivoted.fillna("", inplace=True) 
    df_pivoted.reset_index(drop=True, inplace=True)
    						
    df_pivoted = df_pivoted[['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']]
    return df_pivoted

given_user_df = user_input()
n_users = 5  
n_exercises = 5  

if st.button("Get Weekly Schedule"):
    weekly_schedule,ex_info = FinalRecommenderSystem(given_user_df, n_users, n_exercises)
    df_pivoted = transform_schedule(weekly_schedule)
    
    st.write("Your Weekly Exercise Schedule:")
    st.table(df_pivoted)

    
    st.write("Exercises Info: ")
    st.table(ex_info.reset_index(drop=True))
