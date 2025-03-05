import os
import json
import time
import requests
import paho.mqtt.client as mqtt
import logging
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
API_TOKEN = os.getenv("OP3_API_TOKEN")
SHOW_UUID = "19f08e18-075f-5167-9fbf-00c912b5dba6"
OP3_API_URL = f"https://op3.dev/api/v1/downloads/show/{SHOW_UUID}"

# MQTT Configuration
MQTT_BROKER = os.getenv("MQTT_BROKER", "mqtt.why2025.org")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "podcast/whycast")

# Fetch interval (in seconds)
FETCH_INTERVAL = int(os.getenv("FETCH_INTERVAL", 60))  # Default: 60 seconds

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def fetch_podcast_data():
    """Fetches podcast data from OP3 API."""
    try:
        headers = {"Authorization": f"Bearer {API_TOKEN}"}
        response = requests.get(OP3_API_URL, headers=headers, timeout=10)
        response.raise_for_status()  # Raise an error for HTTP codes 4xx/5xx

        data = response.json()
        logging.info(f"Fetched podcast data: {json.dumps(data, indent=2)}")
        return data
    except requests.exceptions.RequestException as e:
        logging.error(f"Error fetching podcast data: {e}")
        return None

def on_connect(client, userdata, flags, rc):
    """MQTT on-connect callback."""
    if rc == 0:
        logging.info(f"Connected to MQTT broker {MQTT_BROKER}:{MQTT_PORT}")
    else:
        logging.error(f"Failed to connect to MQTT broker, return code {rc}")

def on_disconnect(client, userdata, rc):
    """MQTT on-disconnect callback."""
    logging.warning(f"Disconnected from MQTT broker, reconnecting... ({rc})")
    time.sleep(5)
    client.reconnect()

def publish_to_mqtt(data):
    """Publishes podcast data to the MQTT broker."""
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect

    try:
        client.connect(MQTT_BROKER, MQTT_PORT, 60)
        client.loop_start()
        payload = json.dumps(data)
        client.publish(MQTT_TOPIC, payload)
        logging.info(f"Published to MQTT topic '{MQTT_TOPIC}': {payload}")
    except Exception as e:
        logging.error(f"Failed to publish to MQTT: {e}")
    finally:
        client.loop_stop()
        client.disconnect()

if __name__ == "__main__":
    while True:
        data = fetch_podcast_data()
        if data:
            publish_to_mqtt(data)
        time.sleep(FETCH_INTERVAL)
