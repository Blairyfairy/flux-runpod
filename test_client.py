"""
Simple client to test a deployed RunPod serverless endpoint.

Usage (after setting env vars):
  python test_client.py
"""

import os
import time
import json
import runpod

API_KEY = os.getenv("RUNPOD_API_KEY")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID")

if not API_KEY:
    raise SystemExit("Set RUNPOD_API_KEY environment variable first.")
if not ENDPOINT_ID:
    raise SystemExit("Set RUNPOD_ENDPOINT_ID environment variable first.")

runpod.api_key = API_KEY

TIMEOUT_SECONDS = 600
MAX_RETRIES = 3
POLL_INTERVAL = 3


def load_input_data(file_path="test_input.json"):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
            return data.get("input", data)
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using default payload.")
        return {"prompt": "A cat holding a sign that says hello world"}


def run_job_with_timeout(endpoint, payload, timeout=TIMEOUT_SECONDS):
    job = endpoint.run(payload)
    job_id = getattr(job, "job_id", getattr(job, "id", str(job)))
    print(f"--> Triggered Job ID: {job_id}")

    start_time = time.time()
    while time.time() - start_time < timeout:
        status = job.status()
        elapsed = int(time.time() - start_time)
        print(f"    [{elapsed}s elapsed] Status: {status}")

        if status == "COMPLETED":
            print(f"--> Job {job_id} succeeded!")
            return job.output()
        elif status in ["FAILED", "CANCELLED"]:
            raise RuntimeError(f"Job {job_id} ended with status: {status}")

        time.sleep(POLL_INTERVAL)

    print(f"--> Job {job_id} timed out. Cancelling...")
    try:
        job.cancel()
    except Exception as e:
        print(f"Failed to cancel job: {e}")

    raise TimeoutError(f"Job {job_id} exceeded timeout threshold.")


def execute_with_retries(payload):
    endpoint = runpod.Endpoint(ENDPOINT_ID)

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n[Attempt {attempt}/{MAX_RETRIES}] Submitting request to RunPod Endpoint ({ENDPOINT_ID})...")
        try:
            return run_job_with_timeout(endpoint, payload, timeout=TIMEOUT_SECONDS)
        except (TimeoutError, RuntimeError) as err:
            print(f"Attempt {attempt} failed: {err}")
            if attempt < MAX_RETRIES:
                print("Retrying request...")
            else:
                raise


if __name__ == "__main__":
    payload_input = load_input_data("test_input.json")

    try:
        response = execute_with_retries(payload_input)
        print("\nFinal Output:")
        # Do not dump the full base64 image to the console
        if isinstance(response, dict) and "image_base64" in response:
            img = response.pop("image_base64")
            print(json.dumps(response, indent=2))
            print(f"(image_base64 length: {len(img)} characters)")
            # Optionally save it
            try:
                import base64
                with open("output.png", "wb") as f:
                    f.write(base64.b64decode(img))
                print("Saved image to output.png")
            except Exception as e:
                print(f"Could not save image: {e}")
        else:
            print(json.dumps(response, indent=2))
    except Exception as e:
        print(f"\nExecution failed: {e}")
