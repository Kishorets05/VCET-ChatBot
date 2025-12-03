# 📘 VCET AI Assistant – RAG Chatbot & Timetable Generator

> An end-to-end AI-powered academic assistant built for the VCET CSE Department, combining Retrieval-Augmented Generation (RAG), Google Gemini API, PDF-based document search, and Faculty Timetable Generation.

---

## 🚀 Features

### 🔍 1. RAG Chatbot

- **Document Intelligence**: Extracts information from department PDFs including:
  - 📚 Syllabus
  - 📋 Regulations
  - 📊 Placement reports
  - 📝 Publications
  - 🏢 Infrastructure
  - 👥 Faculty list

- **Advanced Search**: Uses document embeddings + similarity search for accurate retrieval
- **AI-Powered Responses**: Generates accurate answers using Google Gemini API

### 🗂 2. Automated Timetable Generator

- ✅ Identifies subjects for selected semesters
- ✅ Extracts course titles dynamically
- ✅ Generates PDF timetable for faculty
- ✅ Uses fallback PDF extraction if API fails

### 🌐 3. Flask Web Application

- **Modern Interface**: Clean HTML/CSS/JS frontend
- **RESTful APIs** for:
  - 💬 Chatbot queries
  - 📅 Timetable generation
  - 🔧 Initialization
  - 📥 File downloads
- **Stable Backend**: Robust Flask server for ML + RAG logic

---

## 🛠 Tech Stack

| Layer | Technologies |
|-------|-------------|
| **Frontend** | HTML, CSS, JavaScript |
| **Backend** | Flask, Python |
| **AI / ML** | Gemini API, Embeddings |
| **Vector DB** | ChromaDB |
| **PDF Processing** | PyPDF2 |
| **File Storage** | Local File System |
| **Models** | Embedding + Local LLM Helpers |

---

## ▶️ How to Run

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Install packages**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the server**
   ```bash
   python app/web_app.py
   ```

3. **Open in browser**
   ```
   http://localhost:5000
   ```

---

## 📄 Future Enhancements

- [ ] Add authentication (student/faculty login)
- [ ] Deploy on cloud (HuggingFace/Railway/AWS)
- [ ] Add text-to-speech responses
- [ ] Add marks analytics dashboard

---

## 📝 License

This project is developed for VCET CSE Department academic purposes.

---

## 👥 Contributors

Developed for VCET CSE Department

---

**Made with ❤️ for VCET CSE Department**
<!-- Email mapping fixed -->
