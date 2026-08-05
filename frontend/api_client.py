import os
import requests

BASE_URL = os.getenv("API_URL", "http://localhost:8000")
QUERY_URL = f"{BASE_URL.rstrip('/')}/query"
UPLOAD_URL = f"{BASE_URL.rstrip('/')}/upload"


def query_backend(question: str) -> dict:
    """Sends user query to FastAPI and returns dictionary containing 'answer' and 'sources'."""
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(QUERY_URL, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {
            "answer": "Error: Unable to connect to backend server. Ensure FastAPI is running on http://localhost:8000.",
            "sources": []
        }
    except Exception as e:
        return {
            "answer": f"Unexpected Error: {str(e)}",
            "sources": []
        }


def upload_document(uploaded_file) -> dict:
    """Sends an uploaded file from Streamlit to FastAPI's /upload endpoint for ingestion."""
    try:
        files = {
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }
        response = requests.post(UPLOAD_URL, files=files, timeout=120)
        response.raise_for_status()
        return response.json()

    except requests.exceptions.ConnectionError:
        return {
            "error": "Unable to connect to backend server. Ensure FastAPI is running on http://localhost:8000."
        }
    except requests.exceptions.HTTPError as e:
        try:
            error_detail = response.json().get("detail", str(e))
        except Exception:
            error_detail = str(e)
        return {"error": error_detail}
    except Exception as e:
        return {"error": f"Unexpected Error: {str(e)}"}

