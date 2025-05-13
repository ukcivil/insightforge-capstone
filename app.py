import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
import os

# Securely set API key from Streamlit Cloud secrets
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

st.set_page_config(page_title="InsightForge BI Assistant", layout="wide")
st.title("📊 InsightForge – AI-Powered Business Intelligence Assistant")

# Sidebar – Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload Sales Data CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data Loaded")

    # Prepare documents for embedding
    documents = []
    for _, row in df.iterrows():
        doc = {
            "date": pd.to_datetime(row['Date']).strftime('%Y-%m-%d'),
            "product": row['Product'],
            "region": row['Region'],
            "sales": row['Sales'],
            "customer_age": row['Customer_Age'],
            "gender": row['Customer_Gender'],
            "satisfaction": round(row['Customer_Satisfaction'], 2)
        }
        text = f"""
PRODUCT REPORT

Date: {doc['date']}
Product: {doc['product']}
Region: {doc['region']}
Sales: {doc['sales']}
Customer Age: {doc['customer_age']}
Gender: {doc['gender']}
Satisfaction: {doc['satisfaction']}
"""

        documents.append(Document(page_content=text))

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    retriever = vectorstore.as_retriever()
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

    # Query input
    user_query = st.text_input("💬 Ask a business question:")

    if user_query:
        result = qa_chain.invoke({"query": user_query})
        st.subheader("🧠 AI Insight")
        st.write(result["result"])

        # Visual example for Widget A trend
        if "widget a" in user_query.lower() and "sales trend" in user_query.lower():
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.to_period('M').astype(str)
            widget_a = df[df['Product'] == 'Widget A']
            trend = widget_a.groupby('Month')['Sales'].sum().reset_index()
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(data=trend, x='Month', y='Sales', marker='o', ax=ax)
            ax.set_title('Sales Trend for Widget A')
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
else:
    st.warning("Please upload a CSV file to begin.")
