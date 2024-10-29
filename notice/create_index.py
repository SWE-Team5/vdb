from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="1734fc56-9964-4232-a412-50e211980310")

index_name = "skku-notice"

pc.create_index(
    name=index_name,
    #klue/bert-base
    dimension=768,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)