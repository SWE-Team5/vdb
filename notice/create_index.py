from pinecone import Pinecone, ServerlessSpec

def update_index(index_name, dimension, metric):
    pc = Pinecone(api_key="1734fc56-9964-4232-a412-50e211980310")

    pc.delete_index(index_name)

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

update_index("skku-notice", 768, "cosine")