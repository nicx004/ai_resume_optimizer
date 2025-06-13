# 🧠 AI Resume Optimizer

A Streamlit-based web application that analyzes your resume against a job description and provides actionable suggestions to optimize your resume using NLP techniques.

---

## 🚀 Features

- Upload your resume (PDF or DOCX)
- Paste a job description
- Keyword matching using TF-IDF and spaCy
- Highlighted suggestions for improving:
  - Skills alignment
  - Experience match
  - Section clarity
- Export enhanced resume with suggestions

---

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Backend:** Python
- **NLP Libraries:** spaCy, scikit-learn (TF-IDF), PyPDF2, python-docx
- **Others:** Git, GitHub, OpenAI (optional), Pandas

---

## 📦 Installation

```bash
# Clone the repo
git clone https://github.com/nicx004/ai_resume_optimizer.git
cd ai_resume_optimizer

# Create a virtual environment (optional)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
