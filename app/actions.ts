"use server";

import { google } from "@ai-sdk/google";
import { streamText, generateText } from "ai";
import { createStreamableValue } from "ai/rsc";
import { z } from 'zod';
import { revalidatePath } from 'next/cache';

export interface Message {
    role: "user" | "assistant";
    content: string;
}

const conversationHistory: Record<string, Message[]> = {};

export async function streamConversation(history: Message[]) {
    const model = google("models/gemini-1.5-flash");
  
    const { text } = await generateText({
        model: model,
        messages: history,
    });
  
    return {
      messages: history,
      newMessage: text,
    };
}

export async function getConversationHistory(city: string) {
    return conversationHistory[city] || [];
}

export async function translateText() {
}

export async function mainGeminiStream(history: Message[]) {
    const stream = createStreamableValue();
    const model = google("models/gemini-1.5-pro-latest");
  
    // const sanitizeText = (text: string,) => text.replace(/[*_~`]/g, '');
  
    const prompt = `You are a friendly chatbot. Your job is to become the user's best friend. If they
    ask for advice try your best to advise. You MUST finish all responses in complete sentences. NEver just stop in the middle of a sentence.
    
    ${history.map(msg => `${msg.role === "user" ? "User" : "Assistant"}: ${msg.content}`).join('\n')}
    
    `;
    
    (async () => {
      const { textStream } = await streamText({
        model: model,
        messages: [{ role: "user", content: prompt }],
        temperature: 0.9, 
        topP: 0.85,
        topK: 40,
      });
  
      stream.done();
    })().then(() => {});
    
    return {
      messages: history,
      newMessage: stream.value,
    };
  }


