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
import shutil

import langchain
import openai
from langchain import OpenAI
from langchain.chains import LLMChain, RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.document_loaders.unstructured import UnstructuredFileLoader
from langchain.embeddings.openai import OpenAIEmbeddings
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
        self.embedding = OpenAIEmbeddings(
            model="text-embedding-ada-002", max_retries=10
        )
        try:
            self.db = FAISS.load_local(self.store_path, self.embedding, index_name)
        except:
            self.db = None

        if os.path.exists(self.index_hash_path):
            with open(self.index_hash_path, "r") as f:
                self.index_hash = json.load(f)
        else:
            self.index_hash = {index_name: []}

    def is_empty(self):
        return True if self.db is None else False

    def clear_all(self):
        self.db = None
        self.index_hash = {self.index_name: []}
        shutil.rmtree(self.store_path)
        os.mkdir(self.store_path)

    def add_docs(self, base_dir):
        new_docs = []

        if os.path.isfile(base_dir):
            if any(fnmatch.fnmatch(base_dir, p) for p in self.supported_suffixes):
                new_docs.append(base_dir)
        else:
            for root, dirs, files in os.walk(base_dir):
                for matched_file in [
                    f
                    for f in files
                    if any(fnmatch.fnmatch(f, p) for p in self.supported_suffixes)
                ]:
                    new_file_path = os.path.join(root, matched_file)
                    with open(new_file_path, "rb") as f:
                        hash_hex = hashlib.md5(f.read()).hexdigest()
                        if hash_hex not in self.index_hash[self.index_name]:
                            self.index_hash[self.index_name].append(hash_hex)
                            new_docs.append(new_file_path)
                            logger.info(
                                f"Find a new doc, path: {new_file_path}, hash: {hash_hex}."
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


def index_docs(path):
    lfs = LocalFaissStore()
    lfs.add_docs(path)


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
    llm = OpenAI(temperature=0, max_tokens=1500)
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

    print(f"Response: {colored_answer}")


def argument_parser():
    parser = argparse.ArgumentParser(description="GPT Assisstant")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Index-docs command
    summarize_parser = subparsers.add_parser(
        "index-docs",
        help="Index documents, it supports to index a directory or a file.",
    )
    summarize_parser.add_argument(
        "path", type=str, help="File/Directory path to index."
    )

    # Summarize-doc command
    summarize_parser = subparsers.add_parser(
        "summarize-doc", help="Summarize document, it supports lots of file types."
    )
    summarize_parser.add_argument("file_path", type=str, help="File path to summarize.")

    # Query command
    query_parser = subparsers.add_parser("ask", help="Ask question.")
    query_parser.add_argument("query", type=str, help="The question to ask.")

    return parser


def main() -> None:
    args_parser = argument_parser()
    args = args_parser.parse_args()
    if args.command == "index-docs":
        index_docs(args.path)
    elif args.command == "summarize-doc":
        summarize_doc(args.file_path)
    elif args.command == "ask":
        ask(args.query)
    else:
        args_parser.print_help()


if __name__ == "__main__":
    main()
