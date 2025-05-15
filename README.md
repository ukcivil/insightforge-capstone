# InsightForge – AI-Powered Business Intelligence Assistant

InsightForge is a Streamlit-based capstone project that enables non-technical users to derive actionable insights from business data using natural language. Powered by OpenAI's GPT and LangChain’s Retrieval-Augmented Generation (RAG), this tool allows interactive exploration of sales trends, product performance, customer satisfaction, and demographic insights.

---

## Features

- Upload structured CSV sales data
- Ask business questions in natural language
- Get AI-generated answers powered by GPT-3.5
- View relevant charts for trends, satisfaction, and product breakdowns
- Download chat interaction logs for audit or review

---

## File Overview

| File               | Description                                     |
|--------------------|-------------------------------------------------|
| `app.py`           | Main Streamlit application                      |
| `sales_data.csv`   | Sample sales dataset used for testing           |
| `requirements.txt` | Python package dependencies                     |
| `README.md`        | This file – project overview and usage guide    |
| `.gitignore`       | Excludes temp/log files from version control    |
| `.devcontainer/`   | (Optional) VS Code container config             |

---

## Setup Instructions

###  1. Clone the Repository
```bash
git clone https://github.com/YOUR-USERNAME/insightforge-capstone.git
cd insightforge-capstone

2. Install Requirements
Use a virtual environment or run:
pip install -r requirements.txt

3. Launch the App
streamlit run app.py

Example Questions to Try
"Compare Widget A and Widget B sales by region"

"Show the monthly sales trend over time"

"Which product has the highest customer satisfaction?"

"How do customer ages vary by region?"

Downloadable Logs
Each user interaction is logged to chat_log.txt, which can be downloaded via the app interface.

Submission Info
This project meets all capstone requirements for:

Data insight generation

Prompt-based natural language interface

Visualizations

Deployment-ready business tool

Built using:
OpenAI · LangChain · Streamlit · Matplotlib · Seaborn
