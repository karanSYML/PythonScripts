from poelis_sdk import PoelisClient

# Create client
poelis_client = PoelisClient(
    api_key="poelis_live_A1B2C3...",    # API Keys
)

poelis = poelis_client.browser

og3_ws = poelis.og3.og003_technical_data

