import dotenv from "dotenv";
dotenv.config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MistralAIEmbeddings } from "@langchain/mistralai";

// Step 1
const loader = new PDFLoader("./story.pdf");
const data = await loader.load();
// console.log(data)

//Step 3
const embeddings = new MistralAIEmbeddings({
  apiKey: process.env.MISTRAL_API_KEY,
  model: "mistral-embed",
});

// Step 2
const splitter = new RecursiveCharacterTextSplitter({
  chunkSize: 500,
  chunkOverlap: 50,
});

const chunks = await splitter.splitDocuments(data);
// console.log(chunks)

const texts = chunks.map((doc) => doc.pageContent);

const vectors = await Promise.all(
  texts.map(async (chunk) => {
    const embedding = await embeddings.embedQuery(chunk);
    return {
      text: chunk,
      embedding,
    };
  }),
);

console.log(vectors); //1024 Dimensions
