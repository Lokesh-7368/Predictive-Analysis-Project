import argparse
import base64
import json
from pathlib import Path
from urllib import error, request


def parse_args():
    parser = argparse.ArgumentParser(
        description="Send a CT image to the Predictive Analysis Project prediction API."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to the image file to test.",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8080",
        help="Base URL of the running app. Example: http://127.0.0.1:8080",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    image_bytes = image_path.read_bytes()
    payload = json.dumps(
        {
            "image": base64.b64encode(image_bytes).decode("utf-8"),
        }
    ).encode("utf-8")

    api_url = args.url.rstrip("/") + "/predict"
    req = request.Request(
        api_url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Request failed with status {exc.code}: {body}") from exc

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
