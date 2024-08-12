import { NextResponse } from 'next/server';
import { PineconeStore } from '@langchain/pinecone';
import { Pinecone } from '@pinecone-database/pinecone';
import { GoogleGenerativeAIEmbeddings } from '@langchain/google-genai';
import { ChatGoogleGenerativeAI } from '@langchain/google-genai';
import { createRetrievalChain } from 'langchain/chains/retrieval';
import { createStuffDocumentsChain } from 'langchain/chains/combine_documents';
import { PromptTemplate } from '@langchain/core/prompts';
import { Message } from '@/app/chatbot-components/chat';

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
      modelName: 'gemini-1.5-flash',
      apiKey: process.env.GOOGLE_GENERATIVE_AI_API_KEY!,
    });

    // Create prompt template
    const prompt = PromptTemplate.fromTemplate(`
      You are a friendly chatbot that gives the BEST advice for sneakers. You are the BIGGEST sneakerhead. You use every bit of information you know from the Context to provide the best answer and use all available information from the chat history to help determin your answer. Your job is to become the user's best friend. If they ask for advice try your best to advise. You MUST finish all responses in complete sentences. Never just stop in the middle of a sentence.
      Question: {question}
      Context: {context}
      chat_history: {chat_history}
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
      chat_history: history.map((message: Message) => `${message.role} said ${message.content}`).join('\n'),
      question: message,
      context: documentChain.toJSON(),
    });

    return NextResponse.json({ response: response });
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    return NextResponse.json({ message: 'Internal server error' }, { status: 500 });
  }
}