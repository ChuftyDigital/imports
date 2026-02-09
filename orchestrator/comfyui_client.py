"""ComfyUI API client for submitting and monitoring workflow executions.

Connects to the ComfyUI server via REST API and WebSocket to:
1. Submit workflow JSON with dynamically modified parameters
2. Monitor execution progress in real-time
3. Retrieve generated images
4. Handle errors and retries

Designed for the comfyui:latest-5090 Docker environment.
"""

import io
import json
import os
import time
import uuid
import urllib.request
import urllib.parse
from typing import Optional

try:
    import websocket
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False

from .config import PipelineConfig


class ComfyUIClient:
    """Client for interacting with the ComfyUI API."""

    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.base_url = self.config.comfyui_url
        self.client_id = str(uuid.uuid4())
        self._ws = None

    # -------------------------------------------------------------------------
    # REST API
    # -------------------------------------------------------------------------

    def _api_request(self, method: str, endpoint: str,
                     data: Optional[dict] = None) -> dict:
        """Make an API request to ComfyUI."""
        url = f"{self.base_url}{endpoint}"

        if data is not None:
            payload = json.dumps(data).encode("utf-8")
            req = urllib.request.Request(
                url, data=payload, method=method,
                headers={"Content-Type": "application/json"}
            )
        else:
            req = urllib.request.Request(url, method=method)

        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    def queue_prompt(self, workflow: dict) -> str:
        """Submit a workflow to the ComfyUI queue.

        Args:
            workflow: Complete ComfyUI workflow dict (API format)

        Returns:
            prompt_id for tracking execution
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        result = self._api_request("POST", "/prompt", payload)
        return result["prompt_id"]

    def get_history(self, prompt_id: str) -> dict:
        """Get execution history for a prompt."""
        return self._api_request("GET", f"/history/{prompt_id}")

    def get_queue(self) -> dict:
        """Get current queue status."""
        return self._api_request("GET", "/queue")

    def get_image(self, filename: str, subfolder: str = "",
                  img_type: str = "output") -> bytes:
        """Download a generated image from ComfyUI."""
        params = urllib.parse.urlencode({
            "filename": filename,
            "subfolder": subfolder,
            "type": img_type,
        })
        url = f"{self.base_url}/view?{params}"
        with urllib.request.urlopen(url) as response:
            return response.read()

    def upload_image(self, image_path: str, subfolder: str = "",
                     overwrite: bool = True) -> dict:
        """Upload a reference image to ComfyUI's input directory."""
        filename = os.path.basename(image_path)

        with open(image_path, "rb") as f:
            image_data = f.read()

        # Multipart form data
        boundary = uuid.uuid4().hex
        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="image"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: image/png\r\n\r\n"
        ).encode() + image_data + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="subfolder"\r\n\r\n'
            f"{subfolder}\r\n"
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="overwrite"\r\n\r\n'
            f"{'true' if overwrite else 'false'}\r\n"
            f"--{boundary}--\r\n"
        ).encode()

        req = urllib.request.Request(
            f"{self.base_url}/upload/image",
            data=body,
            method="POST",
            headers={
                "Content-Type": f"multipart/form-data; boundary={boundary}",
            },
        )
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode("utf-8"))

    def get_system_stats(self) -> dict:
        """Get system stats including GPU memory usage."""
        return self._api_request("GET", "/system_stats")

    def interrupt(self) -> None:
        """Interrupt the current generation."""
        self._api_request("POST", "/interrupt")

    # -------------------------------------------------------------------------
    # WEBSOCKET MONITORING
    # -------------------------------------------------------------------------

    def connect_ws(self) -> None:
        """Connect to ComfyUI's WebSocket for real-time progress updates."""
        if not HAS_WEBSOCKET:
            return
        ws_url = f"ws://{self.config.comfyui_host}:{self.config.comfyui_port}/ws?clientId={self.client_id}"
        self._ws = websocket.WebSocket()
        self._ws.connect(ws_url)

    def disconnect_ws(self) -> None:
        """Disconnect WebSocket."""
        if self._ws:
            self._ws.close()
            self._ws = None

    def wait_for_completion(self, prompt_id: str,
                            timeout: int = 600,
                            progress_callback=None) -> dict:
        """Wait for a prompt to finish executing.

        Uses WebSocket if available, falls back to polling.

        Args:
            prompt_id: The prompt ID to wait for
            timeout: Maximum seconds to wait
            progress_callback: Optional callback(current_step, total_steps)

        Returns:
            History dict for the completed prompt
        """
        if self._ws or HAS_WEBSOCKET:
            return self._wait_ws(prompt_id, timeout, progress_callback)
        return self._wait_polling(prompt_id, timeout, progress_callback)

    def _wait_ws(self, prompt_id: str, timeout: int,
                 progress_callback) -> dict:
        """Wait using WebSocket for real-time progress."""
        if not self._ws:
            self.connect_ws()

        start = time.time()
        while time.time() - start < timeout:
            try:
                msg = self._ws.recv()
                if isinstance(msg, str):
                    data = json.loads(msg)
                    msg_type = data.get("type", "")

                    if msg_type == "progress" and progress_callback:
                        d = data.get("data", {})
                        progress_callback(d.get("value", 0), d.get("max", 1))

                    if msg_type == "executing":
                        d = data.get("data", {})
                        if d.get("node") is None and d.get("prompt_id") == prompt_id:
                            break

                    if msg_type == "execution_error":
                        d = data.get("data", {})
                        if d.get("prompt_id") == prompt_id:
                            raise RuntimeError(
                                f"ComfyUI execution error: {d.get('exception_message', 'Unknown')}"
                            )
            except websocket.WebSocketTimeoutException:
                continue

        return self.get_history(prompt_id)

    def _wait_polling(self, prompt_id: str, timeout: int,
                      progress_callback) -> dict:
        """Wait by polling the history endpoint."""
        start = time.time()
        while time.time() - start < timeout:
            history = self.get_history(prompt_id)
            if prompt_id in history:
                outputs = history[prompt_id].get("outputs", {})
                if outputs:
                    return history
            time.sleep(2)
        raise TimeoutError(f"Prompt {prompt_id} did not complete in {timeout}s")

    # -------------------------------------------------------------------------
    # HIGH-LEVEL EXECUTION
    # -------------------------------------------------------------------------

    def execute_workflow(self, workflow: dict, timeout: int = 600,
                         progress_callback=None) -> dict:
        """Submit a workflow and wait for results.

        Args:
            workflow: ComfyUI API-format workflow dict
            timeout: Max seconds to wait
            progress_callback: Optional progress callback

        Returns:
            Dict with output images and metadata
        """
        prompt_id = self.queue_prompt(workflow)

        history = self.wait_for_completion(
            prompt_id, timeout=timeout,
            progress_callback=progress_callback,
        )

        # Extract output images
        outputs = history.get(prompt_id, {}).get("outputs", {})
        result = {
            "prompt_id": prompt_id,
            "images": [],
        }

        for node_id, node_output in outputs.items():
            if "images" in node_output:
                for img_info in node_output["images"]:
                    result["images"].append({
                        "filename": img_info["filename"],
                        "subfolder": img_info.get("subfolder", ""),
                        "type": img_info.get("type", "output"),
                        "node_id": node_id,
                    })

        return result

    def save_images(self, result: dict, output_dir: str) -> list:
        """Download and save all images from an execution result.

        Returns list of saved file paths.
        """
        os.makedirs(output_dir, exist_ok=True)
        saved = []

        for img_info in result.get("images", []):
            img_data = self.get_image(
                img_info["filename"],
                subfolder=img_info.get("subfolder", ""),
                img_type=img_info.get("type", "output"),
            )
            output_path = os.path.join(output_dir, img_info["filename"])
            with open(output_path, "wb") as f:
                f.write(img_data)
            saved.append(output_path)

        return saved
