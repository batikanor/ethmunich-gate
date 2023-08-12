import streamlit as st
from web3 import Web3, HTTPProvider
import requests
import ipfsApi
import json
from PIL import Image
from io import BytesIO
from chainlink_feeds.chainlink_feeds import ChainlinkFeeds

from dotenv import load_dotenv
load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ARBISCAN_API_KEY = os.getenv("ARBISCAN_API_KEY")
GATEWAY_BEARER_TOKEN  = os.getenv("GATEWAY_BEARER_TOKEN")
rpc_url = 'https://rpc.eu-central-2.gateway.fm/v4/arbitrum/non-archival/arb1'



# Set up the Streamlit layout
st.title("NFT Data Viewer")

contract_address = st.text_input("Enter Contract Address:", value="0xdf10ff40755ddbc17fa43ee425a41be3dd244f9c")
# contract_address = st.text_input("Enter Contract Address:", value="0xfe8c1ac365ba6780aec5a985d989b327c27670a1")

contract_address = Web3.to_checksum_address(contract_address)
# token_id = st.number_input("Enter Token ID:", value=23982, step=1)
token_id = st.number_input("Enter Token ID:", value=7, step=1)
token_id = int(token_id)
# Set up the connection to your Ethereum node (RPC)
# rpc_url = 'https://arb1.arbitrum.io/'

session = requests.Session()
session.headers.update({"Authorization": f"Bearer {GATEWAY_BEARER_TOKEN}"})
custom_provider = HTTPProvider(endpoint_uri=rpc_url, session=session)
web3 = Web3(custom_provider)


cf = ChainlinkFeeds()
st.write(cf.get_latest_round_data(pair='ETH_USD'))

def get_abi_from_contr_addr(contract_address):
    # ABI_ENDPOINT = f'https://api.etherscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={ETHERSCAN_API_KEY}'
    ABI_ENDPOINT = f'https://api.arbiscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={ARBISCAN_API_KEY}'
    response = requests.get(ABI_ENDPOINT)   
    response_json = response.json()
    abi_json = response_json['result']
    contract_abi = web3.eth.contract(abi=abi_json).abi
    with st.expander("contract_abi"):
        st.write(contract_abi)
    return contract_abi



if contract_address:
    abi_json = get_abi_from_contr_addr(contract_address)
    abi_json = '[{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"approved","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":false,"internalType":"bool","name":"approved","type":"bool"}],"name":"ApprovalForAll","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Transfer","type":"event"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"approve","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"balance","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"getApproved","outputs":[{"internalType":"address","name":"operator","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"owner","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"operator","type":"address"},{"internalType":"bool","name":"_approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes4","name":"interfaceId","type":"bytes4"}],"name":"supportsInterface","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"tokenURI","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"transferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
    #from arbiscan
    # abi_json = '[{"inputs":[{"internalType":"address","name":"_logic","type":"address"},{"internalType":"address","name":"admin_","type":"address"},{"internalType":"bytes","name":"_data","type":"bytes"}],"stateMutability":"payable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"previousAdmin","type":"address"},{"indexed":false,"internalType":"address","name":"newAdmin","type":"address"}],"name":"AdminChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"beacon","type":"address"}],"name":"BeaconUpgraded","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"implementation","type":"address"}],"name":"Upgraded","type":"event"},{"stateMutability":"payable","type":"fallback"},{"inputs":[],"name":"admin","outputs":[{"internalType":"address","name":"admin_","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newAdmin","type":"address"}],"name":"changeAdmin","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"implementation","outputs":[{"internalType":"address","name":"implementation_","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newImplementation","type":"address"}],"name":"upgradeTo","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newImplementation","type":"address"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"upgradeToAndCall","outputs":[],"stateMutability":"payable","type":"function"},{"stateMutability":"payable","type":"receive"}]'
    nft_contract = web3.eth.contract(address=contract_address, abi=abi_json)
    # if st.button("Fetch NFT Data"):
    st.write("aaa")
    # nft_data = nft_contract.functions.tokenURI(token_id).call() 
    st.write(list(nft_contract.functions))

    # nft_data = nft_contract.functions.getNFTData(token_id).call()
    # nft_data = nft_contract.functions.symbol(token_id).call()
    sym = nft_contract.functions.symbol().call()
    name = nft_contract.functions.name().call()
    # Web3ValidationError: Could not identify the intended function with name `tokenURI`, positional arguments with type(s) `` and keyword arguments with type(s) `{}`. Found 1 function(s) with the name `tokenURI`: ['tokenURI(uint256)'] Function invocation failed due to improper number of arguments.
    uri = nft_contract.functions.tokenURI(token_id).call()
    balanceOf = nft_contract.functions.balanceOf(contract_address).call()
    # response = requests.get(token_uri)
    # nft_data = response.json()
    st.write({"NFT Symbol": sym, "NFT Name" : name, "URI": uri, "Balance Of": balanceOf})
    if len(uri) > 0:
        # api = ipfsApi.Client(host='https://ipfs.infura.io', port=5001)
        #OR 
        # api = ipfsApi.Client(host='http://127.0.0.1', port=5001)
        # content = api.get(f'Qm... {uri}') 
        # content = api.get(uri[uri.find(':') + 3 : ]) # + 3 because of ://
        # st.write(content.decode('utf-8'))  # Decode content as UTF-8 and print
        ipfs_hash = uri.replace("ipfs://", "")
        # Define the URL of a public IPFS gateway
        ipfs_gateway_root = "https://ipfs.io/ipfs/"
        ipfs_gateway_url = ipfs_gateway_root + ipfs_hash

        # Send a request to the gateway to retrieve the data
        response = requests.get(ipfs_gateway_url)

        if response.status_code == 200:
            data = response.content
            st.write(data.decode('utf-8'))  # Assuming the data is UTF-8 encoded text
            data = json.loads(data)
            imgUri = data['image']
            im = imgUri.replace("ipfs://", "")
            im_url = ipfs_gateway_root + im
            response = requests.get(im_url)
            image = Image.open(BytesIO(response.content))
            st.image(image, caption='IPFS Image', use_column_width=True)






        else:
            st.write("Failed to retrieve data from IPFS")





