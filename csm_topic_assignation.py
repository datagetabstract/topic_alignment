import random

import streamlit as st
import openai
from pymilvus import Collection
from pymilvus import connections, utility
import tiktoken
import pandas as pd


QA_COLLECTION = "question_answer_openai_emb"
tk_emb = tiktoken.encoding_for_model("text-embedding-ada-002")
st.set_page_config(layout="wide")


def num_emb_tokens(text):
    return len(tk_emb.encode(text))


class EmbeddingModel:
    model_name = None
    min_score = None
    vector_dim = None

    def __init__(self):
        pass

    def __init_subclass__(cls, **kwargs):
        for required in ('model_name', 'min_score', 'vector_dim'):
            if getattr(cls, required) is None:
                raise TypeError(f"Can't instantiate abstract class {cls.__name__} without {required} attribute defined")
        return super().__init_subclass__(**kwargs)

    @classmethod
    def get_embedding(cls, text: str):
        raise NotImplementedError

    @classmethod
    def get_embeddings(cls, texts: list):
        raise NotImplementedError


class OpenAIEmbedding(EmbeddingModel):
    model_name = "openai-text-embedding-ada-002"
    min_score = 0.8
    vector_dim = 1536

    def __init__(self, openai_key):
        super().__init__()
        openai.api_key = openai_key

    @classmethod
    def get_embedding(cls, text: str):
        text = text.replace("\n", " ")
        if num_emb_tokens(text) >= 8190:
            text = text[: int(len(text) / int(num_emb_tokens(text) / 8190 + 1))]
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")['data'][0]['embedding']


class VectorDB:
    """Connect to the vector database, necessary step to interact with a Milvus Collection"""

    def __init__(self, milvus_uri: str, user: str, password: str):
        self.milvus_uri = milvus_uri
        self.user = user
        self.password = password
        self.db_alias = "default"
        self.is_connected = False
        self.connect_to_milvus()

    def connect_to_milvus(self):
        connections.connect(self.db_alias, uri=self.milvus_uri, user=self.user, password=self.password, secure=self.milvus_uri.startswith('https://'))
        self.is_connected = True

    def disconnect_milvus(self):
        connections.disconnect(self.db_alias)

    def collection_available(self, collection_name: str):
        return self.is_connected and utility.has_collection(collection_name)


class MilvusCollection:
    """Control of a generic Milvus (vector database) Collection."""

    def __init__(self, collection_name: str, emb: EmbeddingModel):
        self.collection_name = collection_name
        self.collection = Collection(self.collection_name)
        self.emb = emb

    def milvus_search(self, query: str, doc_type: str, lang: str, topk: int):
        search_vec = self.emb.get_embedding(query)
        search_params = {"metric_type": "IP", "params": {"level": 2}}
        filters_qa = f'language in ["{lang}"] and doc_meta["doc_type"] in ["{doc_type}"]'
        output_fields = ["doc_meta"]
        results = self.collection.search(
            [search_vec],
            anns_field="text_vector",
            param=search_params,
            limit=topk,
            expr=filters_qa,
            output_fields=output_fields
        )[0]
        return results

    def search_collection(self, **kwargs):
        return NotImplementedError


class QuestionAnswerDocsCollection(MilvusCollection):
    """Control the Milvus (vector database) Collection which power the Question Answer AI Assistant."""

    @staticmethod
    def extract_data_from_result(raw_result) -> dict:
        d_meta = raw_result.entity.get('doc_meta')
        return dict(semantic_score=round(raw_result.distance, 4),
                    doc_type=d_meta["doc_type"],
                    item_id=d_meta["item_id"])

    def search_collection(self, query: str, lang: str, doc_type: str, topk=30) -> list:
        results = self.milvus_search(query, doc_type, lang, topk)
        return [self.extract_data_from_result(r) for r in results if (r.distance >= self.emb.min_score)]

    @staticmethod
    def extract_meta_info(raw_result) -> dict:
        d_meta = raw_result.get('doc_meta')
        return dict(doc_id=raw_result['doc_id'],
                    text=d_meta['text'],
                    doc_type=d_meta["doc_type"],
                    item_id=d_meta["item_id"])


openai_key = st.secrets["openai_key"]
milvus_uri = st.secrets["milvus_uri"]
milvus_user = st.secrets["milvus_user"]
milvus_password = st.secrets["milvus_password"]


emb = OpenAIEmbedding(openai_key)
vector_db = VectorDB(milvus_uri, milvus_user, milvus_password)
qa_docs = QuestionAnswerDocsCollection(QA_COLLECTION, emb)


st.title("Alignment Controller")

with st.form("inputs"):
    st.subheader('Your Alignment inputs')
    language = st.selectbox("Database Language", ("en", "fr", "de", "es"))
    topic = st.text_input("What's the topic you are looking for?")
    user = st.text_input("username [tmp input parameter]", "Alessio")
    corp_id = st.text_input("CorporationID", None)
    corp_id = corp_id if corp_id is not None else corp_id
    submitted = st.form_submit_button("Submit")
    if submitted:
        st.write("Similar alignment found:")
        data = {"name": ["name44"] * 2,
                "date": ["2023-07-22"] * 2,
                "topic": [topic + " for leaders"] * 2,
                "title": ["Best summary 1", "another summary"],
                "link": [f"https://www.getabstract.com/en/summary/random-title/4653", f"https://www.getabstract.com/en/summary/random-title/723"],
                "doc_type": ["summary"] * 2,
                "comment": ["Good", ""],
                "CSM user": ["Alessio", "Alessio"],
                "corporationID": [3534, 3534]}
        st.dataframe(pd.DataFrame(data), use_container_width=True)

        st.write("Current suggestion:")
        docs = qa_docs.search_collection(topic, language, "summary")
        for d in docs:
            d['link'] = f"https://www.getabstract.com/{language}/summary/random-title/{d['item_id']}"
        df = pd.DataFrame(docs)
        df["topic"] = topic
        df["title"] = df.item_id  # tmp solution
        df["last_year_trend"] = [[100 * random.random() for i in range(12)] for x in range(df.shape[0])]
        df["include"] = False
        df["comment"] = ""
        df["popularity_index"] = [random.random() for i in range(df.shape[0])]
        df["corporation_percentage_read"] = [random.random()/2. for i in range(df.shape[0])]
        df["alignment_using_it"] = [["name1", "name2"], ["name5"], ["name45"]] + [[] for i in range(df.shape[0] - 3)]
        edited_df = st.data_editor(
            df,
            column_order=["include", "topic", "title", "link", "doc_type", "comment", "semantic_score",
                          "last_year_trend", "popularity_index", "corporation_percentage_read", "alignment_using_it"],
            column_config={
                "comment": st.column_config.TextColumn(width="large"),
                "link": st.column_config.LinkColumn(),
                "last_year_trend": st.column_config.LineChartColumn(y_min=0., y_max=100.),
                "popularity": st.column_config.ProgressColumn(min_value=0., max_value=1.)
            },
            hide_index=True,
            disabled=["topic", "title", "link", "doc_type", "semantic_score", "last_year_trend", "popularity_index",
                      "corporation_percentage_read", "alignment_using_it"],
            num_rows="dynamic",
            use_container_width=True,
            height=40 * df.shape[0]
        )

# TODO: create a save button using on_click, passing the name as unique request
with st.form("save"):
    st.subheader('Save the current alignment')
    name = st.text_input("Unique entry name")
    save = st.form_submit_button("Save to database", type="secondary", on_click=None, kwargs=None)
    if save:
        st.markdown(edited_df)  # save edited_df to database

with st.form("Search in Alignment database"):
    st.subheader('Search for historical alignment')
    name_to_find = st.text_input("Unique entry name", value="dummmy default")
    search = st.form_submit_button("search database", type="secondary", on_click=None, kwargs=None)
    if search:
        data = {"name": [name_to_find],
                "date": ["2023-09-22"],
                "topic": ["burnout"],
                "title": ["Fake Title"],
                "link": [f"https://www.getabstract.com/en/summary/random-title/4653"],
                "doc_type": ["summary"],
                "comment": ["Good"],
                "CSM user": ["Alessio"],
                "corporationID": [3534]}
        st.dataframe(pd.DataFrame(data), use_container_width=True)
