from flask import Flask, request, jsonify
from flask_cors import CORS
import replicate

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Initialize Replicate API Client
replicate_client = replicate.Client(api_token="r8_OV4qZGhlbfrPsLZz9XF5FrBhz5rRTCO2CZKYj")

@app.route('/process', methods=['POST'])
def process_image():
    data = request.json
    image_url = data.get('image_url')
    
    if not image_url:
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
        return jsonify({"output": output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
