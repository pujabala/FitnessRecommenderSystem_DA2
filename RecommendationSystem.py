import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

#NOTE : DATA is synthetically generated so answers might be wrong

def RecommenderSystem(given_user_df,user_profiles,workoutfeedback,n_users,n_exercises):
    scaler = MinMaxScaler()
    columns_to_scale = ['Age', 'Height (cm)', 'Weight (kg)', 'BMI']
    user_profiles[columns_to_scale] = scaler.fit_transform(user_profiles[columns_to_scale])
    given_user_df[columns_to_scale] = scaler.transform(given_user_df[columns_to_scale])
    combined_df = pd.concat([user_profiles, given_user_df], ignore_index=True)

    le = LabelEncoder()
    categorical_columns = ['Gender', 'Primary Fitness Goal', 'Secondary Fitness Goal', 'Preferred Activity Type', 'Workout Intensity']

    for col in categorical_columns:
        le.fit(combined_df[col])
        combined_df[col] = le.transform(combined_df[col])

    encoded_user_profiles = combined_df.iloc[:-1].copy()
    encoded_given_user = combined_df.iloc[-1:].copy()

    all_users_df = pd.concat([encoded_user_profiles, encoded_given_user], ignore_index=True)
    similarity_matrix = cosine_similarity(all_users_df)
    similarity_scores = similarity_matrix[-1][:-1]
    most_similar_user_indices = np.argsort(similarity_scores)[::-1][:n_users]
    top_n_similarity_scores = similarity_scores[most_similar_user_indices]
    user_ids = list(encoded_user_profiles.iloc[most_similar_user_indices]['uid'])

    similar_users_feedback = workoutfeedback[workoutfeedback['uid'].isin(user_ids)]
    exercise_plan = similar_users_feedback.groupby('exid').agg({
        'frequency': 'sum',
        'rating': 'mean',
        'motivation_level': 'mean'
    }).reset_index()
    exercise_plan_sorted = exercise_plan.sort_values(by=['frequency', 'rating'], ascending=[False, False])
    recommended_exercises = exercise_plan_sorted.head(n_exercises)

    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    schedule = {day: [] for day in days}
    exercises_list = []
    for i, row in recommended_exercises.iterrows():
        exid = row['exid']
        exercises_list.append(exid)
        frequency = row['frequency']
        if frequency == 'Daily':
            for day in days:
                schedule[day].append(exid)
        elif frequency == '5 times a week':
            selected_days = np.random.choice(days, 5, replace=False)
            for day in selected_days:
                schedule[day].append(exid)
        elif frequency == '3 times a week':
            selected_days = np.random.choice(days, 3, replace=False)
            for day in selected_days:
                schedule[day].append(exid)
        elif frequency == 'Twice a week':
            selected_days = np.random.choice(days, 2, replace=False)
            for day in selected_days:
                schedule[day].append(exid)
    return schedule,exercises_list

def FinalRecommenderSystem(given_user_df,n_users,n_exercises):
    user_profiles = pd.read_csv('user_profiles.csv')
    workoutfeedback = pd.read_csv('workoutfeedback.csv')
    exercises = pd.read_csv('exercises.csv')
    schedule,recommended_exercises = RecommenderSystem(given_user_df,user_profiles,workoutfeedback,n_users,n_exercises)
    for day, exercises_list in schedule.items():
        schedule[day] = [exercises[exercises['ExNo'] == exid]['Title'].values[0] for exid in exercises_list]
    
    schedule_df = pd.DataFrame(list(schedule.items()), columns=["Day", "Exercises"])
    schedule_df['Exercises'] = schedule_df['Exercises'].apply(lambda x: ', '.join(x)) 
    exercises_info = exercises.loc[recommended_exercises,['Title','Type','Equipment', 'Level']]
    return schedule_df,exercises_info
    