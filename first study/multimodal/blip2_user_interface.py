import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import base64
from PIL import Image
from io import BytesIO
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch

# Initialize the BLIP-2 model and processor
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")

model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# List to store up to 5 images and their captions
saved_images = []

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Image Caption Generator", className="text-center"), width=12)
    ]),
    dbc.Row([
        dbc.Col(dcc.Upload(
            id='upload-image',
            children=html.Div(['Drag and Drop or ', html.A('Select an Image')]),
            style={
                'width': '100%', 'height': '60px', 'lineHeight': '60px',
                'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px',
                'textAlign': 'center', 'margin': '10px'
            }
        ), width=12),
    ]),
    dbc.Row(id='output-images'),
], fluid=True)

# Callback to update the output based on the uploaded image
@app.callback(
    Output('output-images', 'children'),
    [Input('upload-image', 'contents')],
    [State('upload-image', 'filename')]
)
def update_output(image_content, image_name):
    if image_content is not None:
        # Decode and process the image
        content_type, content_string = image_content.split(',')
        decoded = base64.b64decode(content_string)
        img = Image.open(BytesIO(decoded))

        # Generate text using the BLIP-2 model
        inputs = processor(images=img, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # Convert image to displayable format
        buffer = BytesIO()
        img.save(buffer, format="PNG")
        encoded_image = base64.b64encode(buffer.getvalue()).decode()

        # Save the image and generated text
        saved_images.append((encoded_image, generated_text))
        if len(saved_images) > 5:
            saved_images.pop(0)  # Keep only the last 5 items

        # Create the layout for saved images and captions
        cols = []
        for encoded_image, text in saved_images:
            cols.append(dbc.Col([
                html.Img(src=f'data:image/png;base64,{encoded_image}', style={'width': '200px'}),
                html.P(text, style={'textAlign': 'center'})
            ], width=2))  # Set column width to control the size of each image

        return cols
    
    return []

if __name__ == '__main__':
    app.run_server(debug=True)
