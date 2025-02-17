# Digital-TA Development Document

## Project Overview

Digital-TA is a system aimed at improving the efficiency of digital teaching assistants by integrating artificial intelligence technology to facilitate efficient interactions between teaching assistants and students.

---

## System Architecture

### 1. Tech Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Database**: MongoDB + Redis
- **Containerization**: Docker + Docker Compose
- **Model**: Ollama + Groq

### 2. System Modules

1. **Question Classification and Answering**:
   - Automatically analyze student questions and map them to relevant knowledge points.
   - Integrate LLMs to generate appropriate responses.

2. **Teaching Progress Tracking**:
   - Record course progress and student interaction logs.

3. **Quiz and Feedback Generation**:
   - Automatically generate course-related quiz questions.
   - Provide test scores and learning suggestions.

4. **Load Balancing and Efficient Operation**:
   - Use Redis as a caching layer to improve query efficiency.
   - Implement load balancing with Nginx and Docker.

---

## Deployment Guide

### 1. Prerequisites

- Docker and Docker Compose
- Python 3.9+
- PostgreSQL database
- Hugging Face API Key

### 2. Environment Variables

Create a `.env` file with the following content:

```env
POSTGRES_USER=your_postgres_user
POSTGRES_PASSWORD=your_postgres_password
POSTGRES_DB=your_database_name
REDIS_HOST=redis
REDIS_PORT=6379
HUGGING_FACE_API_KEY=your_huggingface_key
```

### 3. Start Commands

Run the following commands to deploy:

```bash
git clone https://github.com/hibana2077/Digital-TA.git
cd Digital-TA
docker-compose up -d
```

---

## Development Guide

### 1. Local Development Environment

1. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

2. Start the backend:

   ```bash
   uvicorn app.main:app --reload
   ```

3. Start the frontend:

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

### 2. API Documentation

Use FastAPI's built-in Swagger UI by visiting: http://localhost:8000/docs

---

## Testing Strategy

1. **Unit Tests**:
   - Use `pytest` to validate backend logic.

   ```bash
   pytest tests/
   ```

2. **Integration Tests**:
   - Test API integration between frontend and backend.

3. **Stress Tests**:
   - Use tools like `locust` to validate system stability.

---

## Contribution Guide

1. Fork the repository and submit a Pull Request.
2. Format code before submission:

   ```bash
   black .
   ```

3. Ensure all tests in the CI/CD pipeline pass.

---

## FAQ

1. **API is not accessible**:
   - Check if Docker Compose is running correctly.
   - Verify the environment variables are correctly set.

2. **Redis connection fails**:
   - Ensure the Redis container is running.

---

## Future Plans

- Support multilingual teaching.
- Add data visualization features.
- Optimize model generation efficiency.

---

## Contact

For any issues, contact the project maintainer: [hibana2077](mailto:hibana2077@gmail.com)
