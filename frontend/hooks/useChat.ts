import { useState, useEffect, useRef, useCallback } from "react";
import type { Message } from "@/types";
import { sendChatQuery } from "@/services";

const FALLBACK_MESSAGE =
  "No specific clinical match found. Please review the diagnostic report for detailed activation maps and pathological assessment.";
const ERROR_MESSAGE =
  "Error connecting to the clinical analysis service. Ensure the diagnostic server is running.";

export interface UseChatReturn {
  messages: Message[];
  input: string;
  setInput: React.Dispatch<React.SetStateAction<string>>;
  isLoading: boolean;
  handleSend: () => Promise<void>;
  scrollRef: React.RefObject<HTMLDivElement | null>;
}

export function useChat(): UseChatReturn {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState<string>("");
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  // Auto-scroll to bottom whenever messages change
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = useCallback(async () => {
    const trimmed = input.trim();
    if (!trimmed || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: trimmed,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const data = await sendChatQuery(userMessage.content);

      const botContent =
        data.documents && data.documents.length > 0
          ? data.documents[0].text
          : FALLBACK_MESSAGE;

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: botContent,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages((prev) => [
        ...prev,
        {
          id: (Date.now() + 1).toString(),
          role: "assistant",
          content: ERROR_MESSAGE,
          timestamp: new Date(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }, [input, isLoading]);

  return { messages, input, setInput, isLoading, handleSend, scrollRef };
}
