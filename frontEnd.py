import requests
import gradio as gr

def call_swagger_api(input_data):
    '''
    Post request on the Swagger API
    '''
    api_url = f'http://localhost:8000/predict-review?review={input_data}'
    
    # Make a POST request to the Swagger API endpoint
    response = requests.post(api_url)
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        result = response.json()
        return result
    else:
        return f"Error"
    
iface = gr.Interface(
    fn=call_swagger_api,
    inputs=["text","text"],
    outputs="text",
    live=False
)

iface.launch()