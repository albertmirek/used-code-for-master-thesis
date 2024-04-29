import requests
import json

def test_torchserve():
    url = "http://localhost:8080/predictions/model"
    headers = {"Content-Type": "application/json"}
    data = {
        "data": [
            {"user_id": None, "product_id": "1026166", "brand_id": "6272", "product_type_id": "63353935393838362D613937362D3131", "customer_price_cz": 399.0, "rating_lifetime": 5.428571429},
            {"user_id": None, "product_id": "1017428", "brand_id": "6804", "product_type_id": "36616661666430362D393162352D3131", "customer_price_cz": 999.0, "rating_lifetime": 9.576923077},
            {"user_id": None, "product_id": "1017424", "brand_id": "6804", "product_type_id": "36616661666430362D393162352D3131", "customer_price_cz": 1299.0, "rating_lifetime": 9.515151515}
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    # Check if the status code is 200 OK
    assert response.status_code == 200, f"Expected status code 200, but got {response.status_code}"

    print("Test passed, response received successfully.")

if __name__ == "__main__":
    test_torchserve()
