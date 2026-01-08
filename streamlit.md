
```markdown
# Streamlit в Google Colab
```bash
!pip install -q streamlit
!npm install localtunnel

# 2 Создаём файл приложения
%%writefile app.py
import streamlit as st
st.title("Моё ML-приложение")

# 3 узнаём наш публичный IP-адрес
!wget -q -0 -ipv4.icanhazip.com

# 3 Запуск приложения в фоне
!streamlit run app.py &>content/logs.txt &

# 3 запуск localtunnel

!npx localtunnel --port 8501

# 4 открываем ссылку (в выводе команды localtunnel появится адрес)

# 5 вводим IP из шага 3