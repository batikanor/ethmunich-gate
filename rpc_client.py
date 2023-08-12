import streamlit as st
from web3 import Web3, HTTPProvider
import requests
import json
from PIL import Image
from io import BytesIO
from chainlink_feeds.chainlink_feeds import ChainlinkFeeds
import os
import re
from ml_utils import get_embedding, get_image_embedding
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from ml_utils import reduce_dimensions
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import DBSCAN
from sklearn.cluster import OPTICS
from sklearn.mixture import GaussianMixture



load_dotenv()

ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")
ARBISCAN_API_KEY = os.getenv("ARBISCAN_API_KEY")
GATEWAY_BEARER_TOKEN  = os.getenv("GATEWAY_BEARER_TOKEN")
rpc_url = 'https://rpc.eu-central-2.gateway.fm/v4/arbitrum/non-archival/arb1'
st.set_page_config(layout="wide")
st.title(':female-detective:    Nft _Similarity_ :blue[Detective]    :male-detective:')
st.divider()
if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = dict()

if 'img_embeddings' not in st.session_state:
    st.session_state['img_embeddings'] = dict()

if 'names' not in st.session_state:
    st.session_state['names'] = dict()

col1, col2, col3 = st.columns([6,8,10])
with col1:
    st.header("Main")
with col2:
    st.header("Metadata")
with col3:
    st.header("Image")



def plot_embeddings_2d(embeddings_2d, keys=None, labels=None):
    fig, ax = plt.subplots()

    # If labels are provided, use them for coloring
    if labels is not None:
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            idxs = [idx for idx, val in enumerate(labels) if val == label]
            ax.scatter(embeddings_2d[idxs, 0], embeddings_2d[idxs, 1], label=label)
    else:
        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
        # Annotate each point with its key if keys are provided
    if keys is not None:
        for idx, key in enumerate(keys):
            ax.annotate(key, (embeddings_2d[idx, 0], embeddings_2d[idx, 1]))

    ax.set_title("2D visualization of embeddings")
    if labels is not None:
        ax.legend()

    st.pyplot(fig)



def increment_string(s):
    # Find all the numbers in the string
    numbers = re.findall(r'\d+', s)
    if not numbers:
        return s
    
    # Take the last number
    last_number = numbers[-1]
    incremented_number = str(int(last_number) + 1)
    
    # Replace only the last occurrence of the number with incremented number
    s = s[::-1].replace(last_number[::-1], incremented_number[::-1], 1)[::-1]
    return s

def decrement_string(s):
    # Find all the numbers in the string
    numbers = re.findall(r'\d+', s)
    if not numbers:
        return s

    # Take the last number
    last_number = numbers[-1]
    decremented_number = str(int(last_number) - 1) if int(last_number) > 0 else "0"

    # Replace only the last occurrence of the number with decremented number
    s = s[::-1].replace(last_number[::-1], decremented_number[::-1], 1)[::-1]
    return s

# Test
# s = "hsfdks80aa"
# print(increment_string(s))  # Expected: hsfdks90aa
# print(decrement_string(s))  # Expected: hsfdks88aa
# Set up the Streamlit layout
with col1:
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
with col2:
    with st.expander("Current Ethereum Price Query"):
        st.write(cf.get_latest_round_data(pair='ETH_USD'))

def get_abi_from_contr_addr(contract_address):
    # ABI_ENDPOINT = f'https://api.etherscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={ETHERSCAN_API_KEY}'
    ABI_ENDPOINT = f'https://api.arbiscan.io/api?module=contract&action=getabi&address={contract_address}&apikey={ARBISCAN_API_KEY}'
    response = requests.get(ABI_ENDPOINT)   
    response_json = response.json()
    abi_json = response_json['result']
    contract_abi = web3.eth.contract(abi=abi_json).abi
    # with col2:
    #     with st.expander("Contract ABI"):
    #         st.write(contract_abi)
    return contract_abi



if contract_address:
    # UNCOMMENTED DUE TO API LIMIT REACHED
    # abi_json = get_abi_from_contr_addr(contract_address)
    abi_json = '[{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"approved","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Approval","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"owner","type":"address"},{"indexed":true,"internalType":"address","name":"operator","type":"address"},{"indexed":false,"internalType":"bool","name":"approved","type":"bool"}],"name":"ApprovalForAll","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"from","type":"address"},{"indexed":true,"internalType":"address","name":"to","type":"address"},{"indexed":true,"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"Transfer","type":"event"},{"inputs":[{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"approve","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"}],"name":"balanceOf","outputs":[{"internalType":"uint256","name":"balance","type":"uint256"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"getApproved","outputs":[{"internalType":"address","name":"operator","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"owner","type":"address"},{"internalType":"address","name":"operator","type":"address"}],"name":"isApprovedForAll","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"name","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"ownerOf","outputs":[{"internalType":"address","name":"owner","type":"address"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"safeTransferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"operator","type":"address"},{"internalType":"bool","name":"_approved","type":"bool"}],"name":"setApprovalForAll","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"bytes4","name":"interfaceId","type":"bytes4"}],"name":"supportsInterface","outputs":[{"internalType":"bool","name":"","type":"bool"}],"stateMutability":"view","type":"function"},{"inputs":[],"name":"symbol","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"tokenURI","outputs":[{"internalType":"string","name":"","type":"string"}],"stateMutability":"view","type":"function"},{"inputs":[{"internalType":"address","name":"from","type":"address"},{"internalType":"address","name":"to","type":"address"},{"internalType":"uint256","name":"tokenId","type":"uint256"}],"name":"transferFrom","outputs":[],"stateMutability":"nonpayable","type":"function"}]'
    with col2:
        with st.expander("Contract ABI"):
            st.write(abi_json)
    #from arbiscan
    # abi_json = '[{"inputs":[{"internalType":"address","name":"_logic","type":"address"},{"internalType":"address","name":"admin_","type":"address"},{"internalType":"bytes","name":"_data","type":"bytes"}],"stateMutability":"payable","type":"constructor"},{"anonymous":false,"inputs":[{"indexed":false,"internalType":"address","name":"previousAdmin","type":"address"},{"indexed":false,"internalType":"address","name":"newAdmin","type":"address"}],"name":"AdminChanged","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"beacon","type":"address"}],"name":"BeaconUpgraded","type":"event"},{"anonymous":false,"inputs":[{"indexed":true,"internalType":"address","name":"implementation","type":"address"}],"name":"Upgraded","type":"event"},{"stateMutability":"payable","type":"fallback"},{"inputs":[],"name":"admin","outputs":[{"internalType":"address","name":"admin_","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newAdmin","type":"address"}],"name":"changeAdmin","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[],"name":"implementation","outputs":[{"internalType":"address","name":"implementation_","type":"address"}],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newImplementation","type":"address"}],"name":"upgradeTo","outputs":[],"stateMutability":"nonpayable","type":"function"},{"inputs":[{"internalType":"address","name":"newImplementation","type":"address"},{"internalType":"bytes","name":"data","type":"bytes"}],"name":"upgradeToAndCall","outputs":[],"stateMutability":"payable","type":"function"},{"stateMutability":"payable","type":"receive"}]'
    nft_contract = web3.eth.contract(address=contract_address, abi=abi_json)
    # if st.button("Fetch NFT Data"):
    # nft_data = nft_contract.functions.tokenURI(token_id).call() 
    with col2:
        with st.expander("Contract Functions"):
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
    with col2:
        with st.expander("Extra NFT Information"):
            st.write({"NFT Symbol": sym, "Collection Name" : name, "URI": uri})
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

            data = json.loads(data)
            imgUri = data['image']
            im = imgUri.replace("ipfs://", "")
            im_url = ipfs_gateway_root + im

            with col2:
                with st.expander("Description"):
                    resp_emb = get_embedding(' '.join([str(item) for item in response.content]))
                    st.session_state['embeddings'][im_url] = resp_emb
                    st.session_state['names'][im_url] = data['name']
                    st.write(response.content.decode('utf-8'))  # Assuming the data is UTF-8 encoded text

            with st.container():
                try:
                    response = requests.get(im_url)
                    image = Image.open(BytesIO(response.content))
                    imemb = get_image_embedding(image)

                    # st.session_state['img_embeddings'][im_url] = image
                    st.session_state['img_embeddings'][im_url] = imemb

                    
                    with col3:
                        st.image(image, caption=data['name'], use_column_width=True)
                        col31, col32, col33 = st.columns([10,1,10])
                        with col33:
                            try:
                                response = requests.get(ipfs_gateway_root + increment_string(im))
                                image = Image.open(BytesIO(response.content))
                                st.image(image, caption='Next NFT', use_column_width=True)
                            except:
                                st.write("There's no next NFT")
                        with col31:
                            try:
                                # st.write(ipfs_gateway_root + decrement_string(im))
                                response = requests.get(ipfs_gateway_root + decrement_string(im))
                                image = Image.open(BytesIO(response.content))
                                st.image(image, caption='Previous NFT', use_column_width=True)
                            except:
                                st.write("There's no previous NFT")
                except Exception as err:
                    with st.expander("NFT couldn't be plotted."):
                        st.write(err)


            with col2:
                with st.expander("Description Embeddings"):
                    for k,v in st.session_state['embeddings'].items():
                        if st.checkbox(f"Show D. Emb.: {k}"):
                            st.write(v)
                with st.expander("Image Embeddings"):
                    for k,v in st.session_state['img_embeddings'].items():
                        if st.checkbox(f"Show  I. Emb.: {k}"):
                            st.write(v)




        else:
            st.write("Failed to retrieve data from IPFS")




emb_ct = len(st.session_state['embeddings'].items())
if emb_ct > 1:
    embeddings = list(st.session_state['embeddings'].values())
    embeddings_2d = reduce_dimensions(embeddings, method='PCA')

    def cluster_embeddings(embeddings, n_clusters=min(3, emb_ct), method="KMEANS", eps=0.5):
        if method == "KMEANS":
            kmeans = KMeans(n_clusters=n_clusters)
            return kmeans.fit_predict(embeddings)
        elif method == "AGGL":
            agglom = AgglomerativeClustering(n_clusters=n_clusters)
            return agglom.fit_predict(embeddings)
        elif method == "DBSCAN":
            dbscan = DBSCAN(eps=eps, min_samples=n_clusters)
            return dbscan.fit_predict(embeddings)
        elif method == "OPTICS":
            optics = OPTICS(min_samples=n_clusters)
            return optics.fit_predict(embeddings)
        elif method == "GMM":
            gmm = GaussianMixture(n_components=n_clusters)
            return gmm.fit_predict(embeddings)



# Plot in Streamlit

    emb_keys_list = list(st.session_state['names'].values())
 
    img_embeddings = list(st.session_state['img_embeddings'].values())
    img_embeddings_array = np.array(img_embeddings)
    # img_embeddings_reshaped = img_embeddings_array.reshape(img_embeddings_array.shape[0], -1)
    img_embeddings_reshaped = img_embeddings_array
    img_embeddings_2d = reduce_dimensions(img_embeddings_reshaped, method='PCA')

    labels = cluster_embeddings(embeddings_2d)
    with st.container():
        cola00,colad0,cola20 = st.columns([3,2,3])
        with cola00:
            st.write("For Description Embeddings")
        with colad0:
            st.write("The Used Method")
        with cola20:
            st.write("For Image Embeddings")

    with st.container():
        cola01,colad1,cola21 = st.columns([3,2,3])
        with cola01:
            labels = cluster_embeddings(embeddings, method="KMEANS")
            plot_embeddings_2d(embeddings_2d,keys=emb_keys_list, labels=labels)
        with colad1:
            st.write("KMeans Clustering: KMeans is an iterative clustering method that partitions data into k distinct clusters based on distance to the nearest centroid. It involves repeatedly assigning points to clusters and recalculating centroids until convergence. One challenge is the need to pre-specify k, and results can vary based on initial centroid placement.")
        with cola21:
            labels = cluster_embeddings(img_embeddings_reshaped, method="KMEANS")
            plot_embeddings_2d(img_embeddings_2d, keys=emb_keys_list, labels=labels)


    with st.container():
        cola02,colad2,cola22 = st.columns([3,2,3])
        with cola02:
            labels = cluster_embeddings(embeddings, method="AGGL")
            plot_embeddings_2d(embeddings_2d,keys=emb_keys_list, labels=labels)
        with colad2:
            st.write("Agglomerative Hierarchical Clustering: This method builds a hierarchy of clusters by successively merging or splitting groups. The dendrogram is a useful tool for understanding the hierarchy.")
        with cola22:
            labels = cluster_embeddings(img_embeddings_reshaped, method="AGGL")
            plot_embeddings_2d(img_embeddings_2d, keys=emb_keys_list, labels=labels)



    with st.container():
        cola03,colad3,cola23 = st.columns([3,2,3])
        with cola03:
            labels = cluster_embeddings(embeddings, method="DBSCAN")
            plot_embeddings_2d(embeddings_2d,keys=emb_keys_list, labels=labels)
        with colad3:
            st.write("DBSCAN (Density-Based Spatial Clustering of Applications with Noise): Groups together points that are close to each other based on a distance measure (usually Euclidean) and a minimum number of points. It also marks as outliers the points that are in low-density regions.")
        with cola23:
            labels = cluster_embeddings(img_embeddings_reshaped, method="DBSCAN")
            plot_embeddings_2d(img_embeddings_2d, keys=emb_keys_list, labels=labels)


    with st.container():
        cola04,colad4,cola24 = st.columns([3,2,3])
        with cola04:
            labels = cluster_embeddings(embeddings, method="OPTICS")
            plot_embeddings_2d(embeddings_2d,keys=emb_keys_list, labels=labels)
        with colad4:
            st.write("OPTICS (Ordering Points To Identify the Clustering Structure): Similar to DBSCAN but doesn’t require setting an eps value. It instead requires setting a minimum number of points to form a cluster.")
        with cola24:
            labels = cluster_embeddings(img_embeddings_reshaped, method="OPTICS")
            plot_embeddings_2d(img_embeddings_2d, keys=emb_keys_list, labels=labels)


    with st.container():
        cola05,colad5,cola25 = st.columns([3,2,3])
        with cola05:
            labels = cluster_embeddings(embeddings, method="GMM")
            plot_embeddings_2d(embeddings_2d,keys=emb_keys_list, labels=labels)
        with colad5:
            st.write("Gaussian Mixture Models (GMM): Assumes that the data is generated from a mixture of several Gaussian distributions. Can be viewed as a generalization of k-means that incorporates information about the covariance structure of the data.")

        with cola25:
            labels = cluster_embeddings(img_embeddings_reshaped, method="GMM")
            plot_embeddings_2d(img_embeddings_2d, keys=emb_keys_list, labels=labels)






