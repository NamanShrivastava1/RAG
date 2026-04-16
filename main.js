import dotenv from "dotenv";
dotenv.config();

import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { MistralAIEmbeddings } from "@langchain/mistralai";
import { Pinecone } from "@pinecone-database/pinecone";

// Step 1
// const loader = new PDFLoader("./story.pdf");
// const data = await loader.load();
// // console.log(data)

// Step 2
// const splitter = new RecursiveCharacterTextSplitter({
//   chunkSize: 500,
//   chunkOverlap: 50,
// });

// const chunks = await splitter.splitDocuments(data);
// // console.log(chunks)

// const texts = chunks.map((doc) => doc.pageContent);

// Step 3
const embeddings = new MistralAIEmbeddings({
  apiKey: process.env.MISTRAL_API_KEY,
  model: "mistral-embed",
});

// Step 4
const pc = new Pinecone({
  apiKey: process.env.PINECONE_API_KEY,
});

const index = pc.index("rag");

// const docs = await Promise.all(
//   texts.map(async (chunk) => {
//     const embedding = await embeddings.embedQuery(chunk);
//     return {
//       text: chunk,
//       embedding,
//     };
//   }),
// );
// // console.log(docs); //1024 Dimensions

// Once Inserted in Pinecone, that's why not using again
// const results = await index.upsert({
//   records: docs.map((doc, i) => ({
//     id: `doc-${i}`,
//     values: doc.embedding,
//     metadata: {
//       text: doc.text,
//     },
//   })),
// });
// // console.log(results)


// Querying Pinecone
const queryEmbedding = await embeddings.embedQuery(
  "How was the internship experience ?",
);
// console.log(queryEmbedding);

const result = await index.query({
  vector: queryEmbedding,
  topK: 2,
  includeMetadata: true,
});

console.log(JSON.stringify(result));
