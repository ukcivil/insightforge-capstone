#  InsightForge – AI-Powered Business Intelligence Assistant

**InsightForge** is a tool built with **Streamlit** that allows the user to explore provided sales data in CSV format using **AI**. It uses **ChatGPT (GPT-3.5)** and **LangChain** to answer questions, create summaries, and generate charts (where relevant) — all from the provided CSV file uploaded into the interface.  The tool also provides a log of questions and answers that can be easily downloaded from within the app to show previous questions and answers.

You can ask questions like:
- “How are sales trending over time?”
- “Which product performs best?”
- “What’s the average customer satisfaction by region?”

---

##  What This App Can Do

- Upload a provided sales CSV file
- Ask questions about the data in English
- Get helpful answers powered by AI and other tools
- See useful charts when relevant
- Download a log of your questions and answers

---

##  What’s in This Project

| File Name          | What It Does                                      |
|--------------------|---------------------------------------------------|
| `app.py`           | The main application code (what Streamlit runs)   |
| `sales_data.csv`   | Sample data provided for testing the app          |
| `requirements.txt` | List of packages needing installation             |
| `README.md`        | This file – explains the project and how to use it|
| `.gitignore`       | Tells Git what files to skip (like logs)          |
| `.devcontainer/`   | Extra config for VS Code (optional)               |

---

##  How to Use It

### 1. Clone This Repository

Open your terminal and type:

```bash
git clone https://github.com/ukcivil/insightforge-capstone.git
cd insightforge-capstone
```

### 2. Install the Required Packages

Ensure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### 3. Run the App

Start the app by typing:

```bash
streamlit run app.py
```

A web page will open where you can upload a CSV and start asking questions.

---

##  Try Asking Questions or Providing Statements Like:

- Compare sales between Widget A and Widget B
- Show the sales trend over the months provided
- What is the average customer age by region?
- Which product has the best satisfaction ratings?

---

##  Interaction Log

Every time you ask a question, it gets saved in a file called `chat_log.txt`. You can download it directly from within the app.

---

## 🚀 How This App Was Built and Deployed

### 🧪 Step 1: Developed in Google Colab

The project started as a Jupyter-style notebook in Google Colab. This allowed quick testing of:
- Data loading and transformation using `pandas`
- Summarizing insights with Python
- Connecting to OpenAI's GPT model through LangChain
- Creating sample visualizations using `matplotlib` and `seaborn`

---

### 🗂️ Step 2: Moved to GitHub for Version Control

After testing, the code was moved into a standalone script (`app.py`) and pushed to a GitHub repository. This made it easy to:
- Organize files (data, scripts, logs)
- Track changes over time
- Connect with external tools like Streamlit

---

### 🌐 Step 3: Deployed with Streamlit Cloud

To make the app usable by anyone:
1. The GitHub repo was connected to [Streamlit Cloud](https://streamlit.io/cloud)
2. Streamlit pulled the `app.py` file and required libraries from `requirements.txt`
3. The app was deployed as a live website

This process made it possible to share the project as a web app where users can:
- Upload a CSV
- Ask questions
- View smart insights and charts instantly

---






---

##  What Tools Were Used

This app was built with:
- **OpenAI GPT-3.5** – helps with proper answers
- **LangChain** – allows searching through the data
- **Streamlit** – allows for a web interface
- **Pandas / Seaborn / Matplotlib** – for charts and data handling

---

##  Items that are required based on the problem statement
- Generates Insight into the Available Data
- Prompt-based natural language interface
- Provides Visualizations when relevant
- Easy to share
---
