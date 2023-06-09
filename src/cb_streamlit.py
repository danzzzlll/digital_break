
from fileinput import filename
from operator import index
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

st.title('Результаты переклассификации') ########

file = st.file_uploader("Upload test.csv file")

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

    with open("/app/digital_break/models/clf_model.pkl", 'rb') as file:
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
    

# Apply the function to each row to compute the 'final type' column
    df['final_type'] = df.apply(compute_final_type, axis=1)
    
    st.markdown('Срез датасета с типом переклассификации 1 или 2') #######
    st.write(df[df['target'] > 0]) ###############
    
    
    sub = pd.read_csv("/app/digital_break/data/submission.csv")
    sub['Тип переклассификации'] = df['target']
    sub['Тип обращения итоговый'] = df.final_type
    
    # st.write(sub)
    
    def convert_df(df):
        return df.to_csv(index=False).encode('utf-8')

    st.download_button(
        label="Download predictions",
        data=convert_df(sub),
        file_name='test_predictions.csv'
    )
    
    df_imp = pd.DataFrame({'feature':clf.feature_names_, 'importance':clf.feature_importances_})
    
    st.markdown('Признаки, исполбзующиеся в модели и их важность') #############
    
    chart = alt.Chart(df_imp).mark_bar().encode(
        x='importance', y=alt.Y('feature', sort=alt.EncodingSortField(field="importance", order='descending'))).interactive()
    st.altair_chart(chart, use_container_width=True, theme='streamlit')
    
    
    st.title('Поиск аномалий') ###########
    
    # file_2 = st.file_uploader("Upload another .csv file")
    
    df = pd.read_csv("/app/digital_break/data/train.csv")
    df = process_for_anomalies(df)
    

    st.markdown('График показывает динамику количества заявок определенного типа по часам (интерактивный)') ###############

    threshold = st.slider('Выберите порог по количеству заявок в час', 30, 100, 5)
    # threshold = 40  # set this to a suitable value
    lst = df.content.value_counts().index.values.tolist()
    grouped_df = df[(df.content.isin(lst))].groupby(['content', pd.Grouper(key='application_date', freq='H')]).size()
    burst_activity = grouped_df[grouped_df >= threshold]
    all_activity = grouped_df[grouped_df > 0]
    burst_activity = burst_activity.reset_index(name='count')
    all_activity = all_activity.reset_index(name='count')
    burst_activity = burst_activity.sort_values(by='application_date')
    all_activity = all_activity.sort_values(by='application_date')
    # st.write(all_activity)

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


    with open("/app/digital_break/models/file.pkl", 'rb') as file:
    # with open("models/file.pkl", 'rb') as file:
        X_2d = pickle.load(file)

    with open("/app/digital_break/models/labels.pkl", 'rb') as file:
    # with open("models/labels.pkl", 'rb') as file:
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

    fig.update_layout(title='Аномалии в назавниях заявок (красные точки - аномалии)')
    # st.markdown('По графику видно, что заявки часто отправляются группами, что может быть связано с тем, что они оформляются на весь отдел, либо связаны с инцидентами. Например, если после инцидента у многих работников пропал доступ к определенному ресурсу') #####

    fig.update_traces(textposition='top center')

    st.plotly_chart(fig, use_container_width=True)


    df_no_incidents = df[df['final_type'] == 'Запрос']

    # Get the timestamps that are within the time window before an incident
    incident_timestamps = df[df['final_type'] == 'Инцидент']['application_date']
    time_window = pd.Timedelta(hours=1)
    pre_incident_timestamps = df_no_incidents['application_date'].apply(lambda x: any((incident_timestamps - x <= time_window) & (incident_timestamps - x > pd.Timedelta(minutes=0))))

    # Filter the dataframe by these timestamps
    pre_incident_reports = df_no_incidents[pre_incident_timestamps]

    # Group by timestamp and count the number of reports
    report_counts = df_no_incidents.groupby('application_date')['content'].count()

    # Create the plot with Plotly
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=report_counts.index, y=report_counts, mode='lines', name='Number of Reports'))
    incident_report_counts = report_counts[report_counts.index.isin(incident_timestamps)]
    fig.add_trace(go.Scatter(x=incident_report_counts.index, y=incident_report_counts, mode='markers', name='Incidents', marker_color='red'))

    fig.update_layout(title='Динамика количества заявок относительно инцидентов',
    xaxis_title='Time',
    yaxis_title='Number of Reports')

    # Display the figure with Streamlit
    st.plotly_chart(fig)