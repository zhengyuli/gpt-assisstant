# Copyright (c) 2023 Zhengyu Li
#
# Licensed under the GPL License version 3.0
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ======================================================================

"""AI assisstant based on chatgpt.
"""

import argparse
import fnmatch
import hashlib
import json
import logging
import os
import re
import shutil

import langchain
import openai
from langchain import OpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.input import get_colored_text
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.vectorstores.base import VectorStoreRetriever
from tqdm import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("gpt_assisstant")

langchain.verbose = True
openai.proxy = os.environ.get("http_proxy")


class LocalFaissStore:
    store_path = os.path.expanduser("~/.faiss_store")
    index_hash_path = os.path.join(store_path, "index_hash.json")
    supported_suffixes = ("*.txt", "*.pdf", "*.csv", "*.xls", "*.doc", "*.docx")

    def __init__(self, index_name="default_index"):
        if not os.path.exists(self.store_path):
            os.mkdir(self.store_path)

        self.index_name = index_name
        self.embedding = HuggingFaceEmbeddings()
        try:
            self.db = FAISS.load_local(self.store_path, self.embedding, index_name)
        except:
            self.db = None

        if os.path.exists(self.index_hash_path):
            with open(self.index_hash_path, "r") as f:
                self.index_hash = json.load(f)
        else:
            self.index_hash = {index_name: {}}

    def is_empty(self):
        return True if self.db is None else False

    def clear_all(self):
        self.db = None
        self.index_hash = {self.index_name: {}}
        shutil.rmtree(self.store_path)
        os.mkdir(self.store_path)

    def add_docs(self, base_dir):
        new_docs = []

        all_files = []
        if os.path.isfile(base_dir):
            if any(fnmatch.fnmatch(base_dir, p) for p in self.supported_suffixes):
                all_files.append(base_dir)
        else:
            for root, dirs, files in os.walk(base_dir):
                matched_files = [
                    os.path.join(root, f)
                    for f in files
                    if any(fnmatch.fnmatch(f, p) for p in self.supported_suffixes)
                ]
                all_files += matched_files

        for new_file in all_files:
            with open(new_file, "rb") as f:
                hash_hex = hashlib.md5(f.read()).hexdigest()
                if hash_hex not in self.index_hash[self.index_name]:
                    self.index_hash[self.index_name][hash_hex] = new_file
                    new_docs.append(new_file)
                    logger.info(f"Find a new doc, path: {new_file}, hash: {hash_hex}.")
                else:
                    logger.info(
                        f"Duplicated doc, path: {new_file}, hash: {hash_hex}, this doc has been indexed before."
                    )

        if new_docs:
            logger.info("Indexing begin... .. .!")
            self._index(new_docs)
            logger.info("Indexing end!")
            with open(self.index_hash_path, "w") as f:
                json.dump(self.index_hash, f)
        else:
            logger.info("There is no new doc.")

    def _index(self, docs):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)

        for doc in tqdm(docs):
            loader = UnstructuredFileLoader(doc)
            split_documents = text_splitter.split_documents(loader.load())
            for doc in split_documents:
                logger.info(doc)
            if self.db is None:
                self.db = FAISS.from_documents(split_documents, self.embedding)
            else:
                self.db.add_documents(split_documents)

        self.db.save_local(self.store_path, self.index_name)

    def similarity_search(self, query):
        return self.db.similarity_search_with_relevance_scores(query)


def index_docs(path):
    logger.setLevel(logging.INFO)
    lfs = LocalFaissStore()
    lfs.add_docs(path)


def clear_index():
    lfs = LocalFaissStore()
    lfs.clear_all()


def summarize_doc(file_path):
    if not os.path.isfile(file_path):
        raise ValueError("doesn't support directory, please input a file instead.")

    loader = UnstructuredFileLoader(file_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    split_documents = text_splitter.split_documents(loader.load())

    llm = OpenAI(max_tokens=1500)

    chain = load_summarize_chain(llm, chain_type="refine", verbose=True)
    summary_info = chain.run(split_documents)
    print("\nSummary: {}".format(get_colored_text(summary_info.strip(), "red")))


def ask(query):
    llm = OpenAI(max_tokens=1500)
    lfs = LocalFaissStore()
    print("Question: {question}".format(question=get_colored_text(query, "green")))
    colored_answer = ""
    if not lfs.is_empty():
        retriever = VectorStoreRetriever(vectorstore=lfs.db)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=compression_retriever)
        result = retrievalQA(query)["result"].strip()
        colored_answer = get_colored_text(result, "green")
    else:
        prompt = PromptTemplate.from_template(query)
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        result = llm_chain.predict().strip()
        colored_answer = get_colored_text(result, "green")

    print(f"\nResponse: {colored_answer}\n\n")


def chat():
    llm = OpenAI(max_tokens=1500)
    lfs = LocalFaissStore()
    if not lfs.is_empty():
        retriever = VectorStoreRetriever(vectorstore=lfs.db)
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        retrievalQA = RetrievalQA.from_llm(llm=llm, retriever=compression_retriever)

    print(
        get_colored_text(
            "Chat Mode: type in any questions you want to ask, or exit command to exit chat mode!",
            "yellow",
        )
    )
    while True:
        try:
            query = input("Enter your question: ")
        except KeyboardInterrupt:
            break

        if query == "exit":
            break

        if not lfs.is_empty():
            result = retrievalQA(query)["result"].strip()
            colored_answer = get_colored_text(result, "green")
        else:
            result = llm.predict().strip()
            colored_answer = get_colored_text(result, "green")

        print(f"\nResponse: {colored_answer}\n")

    print(get_colored_text("\nExit chat mode!", "yellow"))


def similarity_search(query):
    lfs = LocalFaissStore()

    contents = sorted(lfs.similarity_search(query), key=lambda x: x[1], reverse=True)
    for index, content in enumerate(contents):
        prefix = get_colored_text(f"Result-{index}", "red")
        page_content = get_colored_text(
            re.sub(r"[\s\n]+", " ", content[0].page_content.strip()), "green"
        )
        score = get_colored_text(f"score: {content[1]}", "yellow")
        print(f"{prefix}: {page_content} -- {score}")


def argument_parser():
    parser = argparse.ArgumentParser(description="GPT Assisstant")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Index docs command
    index_docs_parser = subparsers.add_parser(
        "index-docs",
        help="Index documents, it supports to index a directory or a file.",
    )
    index_docs_parser.add_argument(
        "path", type=str, help="File/Directory path to index."
    )

    # Clear index command
    subparsers.add_parser("clear-index", help="Clear documents index.")

    # Summarize doc command
    summarize_parser = subparsers.add_parser(
        "summarize-doc", help="Summarize document, it supports lots of file types."
    )
    summarize_parser.add_argument("file_path", type=str, help="File path to summarize.")

    # Query command
    query_parser = subparsers.add_parser("ask", help="Ask question.")
    query_parser.add_argument("query", type=str, help="The question to ask.")

    # Chat command
    query_parser = subparsers.add_parser("chat", help="Chat mode.")

    # Similarity search command
    similarity_search_parser = subparsers.add_parser(
        "similarity-search", help="Similarity content search."
    )
    similarity_search_parser.add_argument(
        "query", type=str, help="The query to search."
    )

    return parser


def main() -> None:
    args_parser = argument_parser()
    args = args_parser.parse_args()
    if args.command == "index-docs":
        index_docs(args.path)
    elif args.command == "clear-index":
        clear_index()
    elif args.command == "summarize-doc":
        summarize_doc(args.file_path)
    elif args.command == "ask":
        ask(args.query)
    elif args.command == "chat":
        chat()
    elif args.command == "similarity-search":
        similarity_search(args.query)
    else:
        args_parser.print_help()


if __name__ == "__main__":
    main()
