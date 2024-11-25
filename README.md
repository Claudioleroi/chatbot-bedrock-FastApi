# README: Service Client AI Assistant API

## Overview
The **Service Client AI Assistant API** is a FastAPI application that integrates with AWS Bedrock for generating AI-powered responses to customer queries based on the context of uploaded files (such as PDFs, CSVs, or text files). The API allows file uploads, context-based query predictions, and dynamic logging. It leverages machine learning models and cosine similarity to find relevant file content in response to user queries.

## Features
- **File Upload**: Users can upload CSV, PDF, or TXT files, which are processed to extract text content.
- **Context-based Predictions**: Once files are uploaded, the application can use the file content to generate AI-powered responses using AWS Bedrock based on user queries.
- **Real-time Logging**: The application streams logs to the client in real-time using Server-Sent Events (SSE).
- **Multilingual Support**: Supports both English and French for responses, based on user language preference.

## Requirements
- Python 3.8+
- FastAPI
- Uvicorn
- boto3 (for AWS integration)
- Pandas
- PyMuPDF (for PDF parsing)
- Scikit-learn (for text similarity calculations)
- python-dotenv (for environment variable management)
- psutil (for system monitoring)

### Install Dependencies
To install the required dependencies, run:

```bash
pip install -r requirements.txt
```

### Setup AWS Credentials
Ensure your AWS credentials are set up correctly. You can either:
1. Use `aws configure` to set your credentials.
2. Set the environment variables manually:
   ```bash
   export AWS_ACCESS_KEY_ID="your-access-key-id"
   export AWS_SECRET_ACCESS_KEY="your-secret-access-key"
   export AWS_REGION="us-east-1"
   ```

Alternatively, you can add the credentials directly into a `.env` file in the root of your project, like this:
```plaintext
AWS_ACCESS_KEY_ID=your-access-key-id
AWS_SECRET_ACCESS_KEY=your-secret-access-key
AWS_REGION=us-east-1
BEDROCK_MODEL=us.meta.llama3-2-3b-instruct-v1:0
```

## Environment Variables
- `AWS_ACCESS_KEY_ID`: Your AWS Access Key ID.
- `AWS_SECRET_ACCESS_KEY`: Your AWS Secret Access Key.
- `AWS_REGION`: AWS region for the service (default is `us-east-1`).
- `BEDROCK_MODEL`: The ID of the Bedrock model you want to use (default is `us.meta.llama3-2-3b-instruct-v1:0`).
- `ALLOWED_ORIGINS`: Allowed origins for CORS, default is `*`.

## Running the API
To run the API, use `uvicorn`:

```bash
uvicorn main:app --reload
```

This will start the server on `http://127.0.0.1:8000`.

## API Endpoints

### 1. **POST /predict**
Generates a response from AWS Bedrock based on the provided query and file content.

**Request Body**:
```json
{
  "text": "What is the company's mission?",
  "max_length": 2000,
  "language": "fr",
  "file_names": ["document1.csv", "document2.pdf"]
}
```

**Response**:
```json
{
  "response": "The company's mission is to innovate and lead in the tech industry."
}
```

### 2. **POST /upload-training-data**
Uploads a file (CSV, TXT, or PDF) for processing. The content of the file will be indexed and used to generate AI responses.

**Request Body**:
- `file`: The file to upload.

**Response**:
```json
{
  "message": "File uploaded and processed successfully",
  "filename": "document1.csv",
  "company_name": "Example Company"
}
```

### 3. **GET /logs**
Streams logs in real-time via Server-Sent Events (SSE).

**Response**:
```json
{
  "timestamp": "2024-11-25T14:30:00",
  "level": "INFO",
  "message": "File uploaded and processed successfully",
  "process_id": 12345,
  "thread_id": 1,
  "memory_usage": 50.5
}
```

## File Processing
Uploaded files are processed as follows:
- **CSV**: Parsed into a Pandas DataFrame.
- **TXT**: Split into lines and stored in a DataFrame.
- **PDF**: Extracted text using PyMuPDF and stored in a DataFrame.

### File Content Example
For a **PDF** or **TXT** file, the content is split into lines and indexed for similarity matching.

### Querying Files
The `predict` endpoint uses cosine similarity to match the query with relevant file content. It utilizes **TF-IDF vectorization** to calculate similarity between the query and stored file contents.

## Logging
The application provides detailed logging using the Python `logging` module. Logs are captured and streamed to the client via SSE. Logs include information such as memory usage and timestamp.

## Error Handling
If an error occurs during file upload or prediction, a `500 Internal Server Error` will be returned with a detailed error message.

## Troubleshooting
- **Missing environment variables**: Ensure that your `.env` file is correctly configured with your AWS credentials.
- **AWS Errors**: Check your AWS access key, secret key, and region. Verify your Bedrock model ID.

## License
This project is licensed under the MIT License.

---

With this setup, you should be able to deploy and test the Service Client AI Assistant API on your local machine. Feel free to adjust configurations, especially the AWS credentials, based on your environment.
