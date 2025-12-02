📘 VCET AI Assistant – RAG Chatbot & Timetable Generator

An end-to-end AI-powered academic assistant built for the VCET CSE Department, combining:

Retrieval-Augmented Generation (RAG)

Google Gemini API

PDF-based document search

Faculty Timetable Generation

Flask Web Backend

Clean HTML/CSS/JS Frontend

This application intelligently answers student/faculty queries and dynamically generates timetables using department syllabus and faculty details.

🚀 Features
🔍 1. RAG Chatbot

Extracts information from department PDFs:

Syllabus

Regulations

Placement reports

Publications

Infrastructure

Faculty list

Uses document embeddings + similarity search

Generates accurate answers using Gemini API

🗂 2. Automated Timetable Generator

Identifies subjects for selected semesters

Extracts course titles dynamically

Generates PDF timetable for faculty

Uses fallback PDF extraction if API fails

🌐 3. Flask Web Application

Clean HTML/CSS/JS interface

REST APIs for:

Chatbot queries

Timetable generation

Initialization

File downloads

Stable backend for ML + RAG logic

🛠 Tech Stack
Layer	Technologies
Frontend	HTML, CSS, JS
Backend	Flask, Python
AI / ML	Gemini API, Embeddings
Vector DB	ChromaDB
PDF Processing	PyPDF2
File Storage	Local FS
Models	Embedding + Local LLM Helpers
📁 Project Structure
vcet_chatbot_project/
│
├── app/                     # Flask backend
│   ├── web_app.py
│   ├── rag_system.py
│   ├── timetable_generator.py
│   ├── __init__.py
│
├── data/                    # PDF/Text data for RAG
│
├── static/                  # CSS, JS, images
│
├── templates/               # HTML templates
│
├── vectorstore/             # Chroma vector DB
│
├── models/                  # Embedding + local models
│
├── requirements.txt
├── README.md
└── .gitignore

▶️ How to Run
1. Install packages
pip install -r requirements.txt

2. Start the server
python app/web_app.py

3. Open in browser
http://localhost:5000

📄 Future Enhancements

Add authentication (student/faculty login)

Deploy on cloud (HuggingFace/Railway/AWS)

Add text-to-speech responses

Add marks analytics dashboard