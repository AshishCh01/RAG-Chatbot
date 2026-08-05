import os
import requests

# FastAPI target endpoint URL
API_URL = os.getenv("API_URL", "http://localhost:8000/query")


# def query_backend(question: str) -> str:
#     """Sends user query to the FastAPI backend and retrieves the response."""
#     payload = {"question": question}
#     headers = {"Content-Type": "application/json"}

#     try:
#         response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
#         response.raise_for_status()
#         data = response.json()
#         return data.get("answer", "I couldn't find that information in the uploaded documents.")

#     except requests.exceptions.ConnectionError:
#         return "Error: Unable to connect to the backend server. Ensure FastAPI is running on http://localhost:8000."
#     except requests.exceptions.Timeout:
#         return "Error: Request timed out while waiting for backend response."
#     except requests.exceptions.HTTPError as http_err:
#         return f"HTTP Error: {http_err}"
#     except Exception as e:
#         return f"Unexpected Error: {str(e)}"

def query_backend(question: str) -> dict:
    """Sends user query to FastAPI and returns dictionary containing 'answer' and 'sources'."""
    payload = {"question": question}
    headers = {"Content-Type": "application/json"}

    try:
        response = requests.post(API_URL, json=payload, headers=headers, timeout=60)
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