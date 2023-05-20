
import pandas as pd
import numpy as np
import streamlit as st
import pickle
import string
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.tokenize import sent_tokenize
from catboost import CatBoostClassifier
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

file = st.file_uploader("Upload .csv file")


if file is not None:
    df = pd.read_csv(file)
    # st.write(df)

    
    new_names = {'Содержание':'content', 'Сервис':'service', 'Приоритет':'priority', 'Статус':'status', 'Функциональная группа':'group', 'Крайний срок':'deadline', 'Дата обращения':'application_date', 'Тип обращения на момент подачи':'original_type', 'Тип обращения итоговый':'final_type', 'Решение':'decision', 'Тип переклассификации':'target', 'Дата восстановления':'repair_date', 'Дата закрытия обращения':'closing_date', 'Критичность':'criticality', 'Влияние':'influence', 'Система':'system', 'Место':'place'}


    def clean_text(text):
        text = str(text)
        table = text.maketrans(dict.fromkeys(string.punctuation))
        words = word_tokenize(text.lower().strip().translate(table))
        lemmed = [WordNetLemmatizer().lemmatize(word) for word in words]   

        return " ".join(lemmed)

    def process(df):
        df = df.rename(columns={'Решение             ': 'Решение'})
        df = df.rename(columns=new_names)
        df.deadline = df.deadline.apply(pd.to_datetime)
        df.application_date = df.application_date.apply(pd.to_datetime)
        df['available_hours'] = (df.deadline - df.application_date).dt.total_seconds() // 3600
        to_drop = ['deadline', 'application_date', 'closing_date', 'repair_date']
        df = df.drop(columns=to_drop)
        df.content = df['content'].fillna('')
        df['content_clean'] = df.content.apply(clean_text)
        df['decision_clean'] = decision = df.decision.apply(clean_text)

        return df
    
    def process_for_anomalies(df):
        df = df.rename(columns={'Решение             ': 'Решение'})
        df = df.rename(columns=new_names)
        df.deadline = df.deadline.apply(pd.to_datetime)
        df.application_date = df.application_date.apply(pd.to_datetime)
        df['available_hours'] = (df.deadline - df.application_date).dt.total_seconds() // 3600
        df = df.dropna(subset=['content'])
        df = df.reset_index(drop=True)
        
        return df

    with open('CP\clf_model.pkl', 'rb') as file:
        clf = pickle.load(file)

    df = process(df)
    # st.write(df)
    
    df['target'] = clf.predict(df[clf.feature_names_])
    # st.write(pred)
    
    
    def compute_final_type(row):
        if row['target'] == 0:
            return row['original_type']
        elif row['target'] == 2:
            return 'Инцидент'
        elif row['target'] == 1:
            return 'Запрос'
        else:
            return None
        
    st.write(df)
# Apply the function to each row to compute the 'final type' column
    df['final_type'] = df.apply(compute_final_type, axis=1)
    
    sub = pd.read_csv('CP\submission.csv')
    sub['Тип переклассификации'] = df['target']
    sub['Тип обращения итоговый'] = df.final_type
    
    st.write(sub)
    
    df_imp = pd.DataFrame({'feature':clf.feature_names_, 'importance':clf.feature_importances_})
    
    chart = alt.Chart(df_imp).mark_bar().encode(
        x='importance', y=alt.Y('feature', sort=alt.EncodingSortField(field="importance", order='descending'))).interactive()
    st.altair_chart(chart, use_container_width=True, theme='streamlit')
    
    
    file_2 = st.file_uploader("Upload another .csv file")
    
    if file_2 is not None:
        df = pd.read_csv(file_2)
        df = process_for_anomalies(df)

        threshold = st.slider('Pick a threshold', 5, 100, 5)
        # threshold = 40  # set this to a suitable value
        lst = df.content.value_counts().index.values.tolist()
        grouped_df = df[(df.content.isin(lst))].groupby(['content', pd.Grouper(key='application_date', freq='H')]).size()
        burst_activity = grouped_df[grouped_df >= threshold]
        all_activity = grouped_df[grouped_df > 0]
        burst_activity = burst_activity.reset_index(name='count')
        all_activity = all_activity.reset_index(name='count')
        burst_activity = burst_activity.sort_values(by='application_date')
        all_activity = all_activity.sort_values(by='application_date')
        st.write(all_activity)
        
        fig = px.line(all_activity[all_activity.content.isin(list(burst_activity.content.values))], x='application_date', y='count', color='content')
        fig.update_layout(
            autosize=False,
            height=600,
            width=600,
            xaxis_title='Time',
            yaxis_title='Report Count',
            legend=dict(
                orientation="h",  # Horizontal legend
                yanchor="bottom",  # Anchor legend at the bottom
                y=1.02,  # Place legend slightly above the figure's top
                xanchor="right",  # Anchor legend at the right
                x=1  # Place legend at the figure's right
            )
        )
        fig.update_traces(mode='lines+markers', marker=dict(size=5))
        st.plotly_chart(fig, use_container_width=True)
        
        
        with open('CP/file.pkl', 'rb') as file:
            X_2d = pickle.load(file)

        with open('CP/labels.pkl', 'rb') as file:
            anomaly_labels = pickle.load(file)
            
        fig = go.Figure(data=go.Scatter(
            x = X_2d[:, 0],
            y = X_2d[:, 1],
            mode='markers',  # Линии и маркеры
            text=df['content'].values,  # Подписи
            marker=dict(
            size=8,
            color=['red' if anomaly == -1 else 'blue' for anomaly in anomaly_labels],  # Красные для аномалий, синие для обычных точек
                )
                ))

        fig.update_traces(textposition='top center')

        # st.plotly_chart(fig, use_container_width=True)
        
        names = ['(02.12) Прекращение доступа в Система500', 'Часы.', 
                 'Письмо RandomKKKK', 'Письмо Random4945Заявка', '<Бот-наблюдения> Система110Ведение НСИ . Ухудшение статуса сервиса.  ',
                 '(Копия) testing 123',
                 'Система26',
                 ' АА',
                 'Зеркало, жалюзи',
                 '(22.12) Изменение логического доступа',
                 ' 4эт',
                 ' Система8']
        st.write(names)