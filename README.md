# Streamlit Video Edit

This is a Streamlit application for video editing with AI-powered audio transcription, correction, and replacement.

## Features

- Transcribe audio from videos using AssemblyAI.
- Correct transcription using GROQ.
- Replace video audio with AI-generated speech aligned with word-level timestamps.

## Setup Instructions

### 1. Clone the Repository

First, clone the repository to your local machine:

```bash
git clone https://github.com/kazutokidigaya/streamlit-video-edit.git
cd streamlit-video-edit
```

### 2. Create and Activate a Virtual Environment

On **macOS/Linux**:

```bash
python3 -m venv venv
source venv/bin/activate
```

On **Windows**:

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Make sure you have a `.env` file in the root of your project with the following:

```
GROQ_API_KEY="your-api-key"
ASSEMBLYAI_API_KEY="your-api-key"
```

### 5. Run the Application

```bash
python -m streamlit run app.py
```
