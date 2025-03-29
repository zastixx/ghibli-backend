from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate
import os
import logging
import requests
from supabase import create_client, Client
from io import BytesIO

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize logging
logging.basicConfig(level=logging.INFO)

# Environment Variables
REPLICATE_API_TOKEN = os.getenv("REPLICATE_API_TOKEN")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_BUCKET = "images"  # Change this to your Supabase bucket name

if not all([REPLICATE_API_TOKEN, SUPABASE_URL, SUPABASE_KEY]):
    logging.error("Missing required environment variables")

# Initialize Replicate API Client
replicate_client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

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
        response = replicate_client.run(
            "aaronaftab/mirage-ghibli:166efd159b4138da932522bc5af40d39194033f587d9bdbab1e594119eae3e7f",
            input=input_params
        )

        response_list = list(response)  # Convert generator to list
        logging.info(f"Full response from Replicate API: {response_list}")

        if not response_list:
            raise ValueError("Empty response from Replicate API")

        output_url = response_list[0]  # Extract first image URL

        # Download image
        image_response = requests.get(output_url)
        if image_response.status_code != 200:
            raise ValueError("Failed to download image from Replicate")

        image_data = BytesIO(image_response.content)
        image_filename = f"processed_{os.path.basename(output_url)}"

        # Upload to Supabase Storage
        upload_response = supabase.storage.from_(SUPABASE_BUCKET).upload(image_filename, image_data, {"content-type": "image/webp"})
        
        if "error" in upload_response:
            raise ValueError(f"Supabase Upload Error: {upload_response['error']}")

        # Generate public URL for the image
        supabase_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_BUCKET}/{image_filename}"

        logging.info(f"Image uploaded to Supabase: {supabase_url}")
        return jsonify({"output": supabase_url})

    except Exception as e:
        logging.error(f"Error processing image: {str(e)}", exc_info=True)
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5000))  # Render assigns a dynamic port
    app.run(host='0.0.0.0', port=port)
