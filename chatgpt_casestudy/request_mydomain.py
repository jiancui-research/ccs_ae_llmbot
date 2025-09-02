import requests

def send_request_as_bot():
    # Define headers with bot information
    headers = {
        'User-Agent': 'xxxx-Bot/1.0',
        'From': 'xxxx-bot@example.com'
    }

    # The specific test server URL
    url = "http://149.165.155.198:81/"

    try:
        # Send GET request with custom headers
        response = requests.get(url, headers=headers)
        
        # Check if request was successful
        if response.status_code == 200:
            print("Request successful!")
            print("Response:", response.text)
        else:
            print(f"Request failed with status code: {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    send_request_as_bot()
