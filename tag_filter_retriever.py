import faiss
from langchain.vectorstores import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
import numpy as np

class TagFilteredFAISSRetriever:
    def __init__(self, faiss_store, embedding_model):
        self.faiss_store = faiss_store
        self.embedding_model = embedding_model

    def _filter_by_tag(self, tag):
        """返回 tag 筛选后的 (doc_id, doc) 列表"""
        return [
            (i, doc)
            for i, doc in self.faiss_store.docstore._dict.items()
            if doc.metadata.get("tag") == tag
        ]

    def _build_temp_faiss(self, docs, embeddings):
        """在内存中临时构造一个 FAISS 索引，仅用于本次 tag 检索"""
        

        embeddings = np.array(embeddings).astype("float32")
        dim = embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(embeddings)

        docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
        index_to_docstore_id = {i: i for i in range(len(docs))}

        return FAISS(
            embedding_function=self.embedding_model,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id
        )

    def search(self, query=None, img_src=None, tag=None, top_k=5):
        assert query or img_src, "必须提供 query 文本或 img_src 图像路径之一"

        if query:
            query_embedding = self.embedding_model.embed_query(query)
        elif img_src:
            query_embedding = self.embedding_model.embed_image_query(img_src)

        if tag:
            filtered = self._filter_by_tag(tag)
        else:
            filtered = list(self.faiss_store.docstore._dict.items())

        if not filtered:
            return []

        docs = [doc for _, doc in filtered]
        embeddings = [self.faiss_store.index.reconstruct(i) for i, _ in filtered]
        
        
        sub_store = self._build_temp_faiss(docs, embeddings)

        query_vec = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = sub_store.index.search(query_vec, top_k)

        results = [docs[i] for i in indices[0] if i != -1]  
        similarity_score = float(distances[0][0]) 
        return results, similarity_score
