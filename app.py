from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
import os
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Fetch API token from Render's environment variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")

if not REPLICATE_API_TOKEN:
    logging.error("Missing REPLICATE_API_TOKEN. Make sure it is set in the environment variables.")

# Initialize Replicate API Client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    image_url = data.get('image_url')

    if not image_url:
        logging.error("Image URL is missing in the request")
        return jsonify({"error": "Image URL is required"}), 400

    input_params = {
        "image": image_url,
        "model": "dev",
        "prompt": "Ghibli style photo with detailed rendering",
        "go_fast": True,
        "lora_scale": 1,
        "megapixels": "1",
        "num_outputs": 1,
        "aspect_ratio": "1:1",
        "output_format": "webp",
        "guidance_scale": 10,
        "output_quality": 80,
        "prompt_strength": 0.75,
        "extra_lora_scale": 1,
        "num_inference_steps": 28
    }

    try:
        output = replicate_client.run(
            "aaronaftab/mirage-ghibli:166efd159b4138da932522bc5af40d39194033f587d9bdbab1e594119eae3e7f",
            input=input_params
        )
        logging.info("Image processed successfully")
        return jsonify({"output": output})
    except Exception as e:
        logging.error(f"Error processing image: {str(e)}")
        return jsonify({"error": "An error occurred while processing the image"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Render automatically sets the PORT variable
    app.run(host='0.0.0.0', port=port)
