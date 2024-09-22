from flask import Flask, request, render_template
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms
from models.model import create_model

app = Flask(__name__)
model = create_model()
model.eval()  # Set the model to evaluation mode

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    
    image = Image.open(file)
    # Process image and make prediction
    prediction = process_image(image)
    
    # Save the result map
    prediction.save('results/maps/output_map.png')
    
    return render_template('result.html', image='results/maps/output_map.png')

def process_image(image):
    # Preprocess image and run model prediction
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        output = model(input_tensor)['out'][0]  # Get model output
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return Image.fromarray(output_predictions)

if __name__ == '__main__':
    app.run(debug=True)
