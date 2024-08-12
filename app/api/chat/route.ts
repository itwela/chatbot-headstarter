import { NextResponse } from 'next/server';
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { PromptTemplate } from '@langchain/core/prompts';

export async function POST(request: Request) {
  try {
    const { message, history } = await request.json();

    // Initialize Pinecone
    const pinecone = new Pinecone({
      apiKey: process.env.PINECONE_API_KEY!,
    });

    const index = pinecone.Index(process.env.PINECONE_INDEX_NAME!);

    // Create embeddings
    const embeddings = new GoogleGenerativeAIEmbeddings({
      apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY!,
      modelName: 'embedding-001',
    });

    // Create vector store
    const vectorStore = await PineconeStore.fromExistingIndex(embeddings, { pineconeIndex: index });

    // Initialize ChatGoogleGenerativeAI
    const model = new ChatGoogleGenerativeAI({
      modelName: 'gemini-pro',
      apiKey: process.env.GOOGLE_API_KEY!,
    });

    // Create prompt template
    const prompt = PromptTemplate.fromTemplate(`
      Answer the following question based on the provided context:
      Question: {question}
      Context: {context}
      Answer:
    `);

    // Create document chain
    const documentChain = await createStuffDocumentsChain({
      llm: model,
      prompt,
    });

    // Create retrieval chain
    const retrievalChain = await createRetrievalChain({
      combineDocsChain: documentChain,
      retriever: vectorStore.asRetriever(),
    });

    // Get response from the chain
    const response = await retrievalChain.invoke({
      input: message,
      chat_history: history,
    });

    return NextResponse.json({ response: response.answer });
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    return NextResponse.json({ message: 'Internal server error' }, { status: 500 });
  }
}