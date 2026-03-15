"use client";

import React, { useState, useEffect, useRef } from "react";
import { Send, Info } from "lucide-react";
import { Button } from "@heroui/button";
import { Textarea } from "@heroui/input";
import { ScrollShadow } from "@heroui/scroll-shadow";
import clsx from "clsx";
import { sendChatQuery } from "@/services/chat";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  timestamp: Date;
}

export default function ChatSection() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const data = await sendChatQuery(userMessage.content, 3);

      let botContent = "";
      if (data.documents && data.documents.length > 0) {
        botContent = data.documents[0].text;
      } else {
        botContent =
          "No specific clinical match found. Please review the diagnostic report for detailed activation maps and pathological assessment.";
      }

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: botContent,
        timestamp: new Date(),
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (error) {
      console.error("Chat error:", error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content:
          "Error connecting to the clinical analysis service. Ensure the diagnostic server is running.",
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-100px)] max-w-4xl w-full mx-auto bg-background animate-in fade-in duration-500">
      {/* Minimal Header */}
      <header className="px-4 py-4 border-b border-default-100 flex items-center justify-between" />

      {/* Focused Chat Area */}
      <ScrollShadow
        className="flex-1 px-4 py-8 space-y-10 [-ms-overflow-style:none] [scrollbar-width:none] [&::-webkit-scrollbar]:hidden"
        ref={scrollRef}
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6 animate-in fade-in duration-700">
            <p className="text-xs font-bold text-default-500 uppercase tracking-wider">
              Awaiting clinical inquiry
            </p>
            <div className="flex flex-wrap justify-center gap-3 max-w-lg mt-2">
              {[
                "Summarize key radiograph findings",
                "Explain cavitation characteristics",
                "Assess miliary spread probability",
                "Interpret activation maps",
              ].map((query, i) => (
                <button
                  key={i}
                  onClick={() => setInput(query)}
                  className="px-4 py-2 cursor-pointer rounded-full border border-default-200 bg-white text-[11px] font-medium text-default-600 hover:border-primary/40 hover:bg-default-50 transition-all shadow-sm"
                >
                  {query}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((msg) => (
            <div
              key={msg.id}
              className={clsx(
                "flex gap-5 max-w-3xl mx-auto animate-in fade-in slide-in-from-bottom-2 duration-300",
                msg.role === "user" ? "flex-row-reverse" : "flex-row"
              )}
            >
              <div
                className={clsx(
                  "flex-1 space-y-1.5",
                  msg.role === "user" ? "text-right" : "text-left"
                )}
              >
                <div
                  className={clsx(
                    "text-[14px] leading-relaxed font-medium",
                    msg.role === "user"
                      ? "bg-default-100 text-default-900 py-2.5 px-4 rounded-2xl border border-default-200 inline-block"
                      : "text-default-900 py-1"
                  )}
                >
                  {msg.content}
                </div>
                <p className="text-[9px] font-medium text-default-500 uppercase tracking-tight px-1">
                  {msg.timestamp.toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                  })}
                </p>
              </div>
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex gap-5 max-w-3xl mx-auto">
            <div className="flex-1 pt-3.5 space-y-2">
              <div className="h-1.5 w-full bg-default-50 rounded-full animate-pulse"></div>
              <div className="h-1.5 w-2/3 bg-default-50 rounded-full animate-pulse"></div>
            </div>
          </div>
        )}
      </ScrollShadow>

      <footer className="p-4 bg-background">
        <div className="max-w-3xl mx-auto">
          <div className="relative bg-white border border-default-200 rounded-xl focus-within:border-primary/50 focus-within:shadow-md transition-all">
            <Textarea
              placeholder="Clinical inquiry..."
              className="min-h-[44px] max-h-40"
              variant="flat"
              value={input}
              onChange={(e: React.ChangeEvent<HTMLTextAreaElement>) =>
                setInput(e.target.value)
              }
              onKeyDown={(e: React.KeyboardEvent<HTMLTextAreaElement>) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  handleSend();
                }
              }}
              classNames={{
                inputWrapper:
                  "bg-transparent shadow-none p-0 min-h-[44px] hover:bg-transparent focus-within:bg-transparent",
                input:
                  "text-[14px] font-medium pl-4 pr-12 py-3 placeholder:text-default-400 overflow-y-auto",
              }}
            />
            <Button
              isIconOnly
              color="primary"
              radius="lg"
              className="absolute bottom-2 right-2 h-8 w-8 min-w-8 shadow-sm bg-primary"
              onPress={handleSend}
              disabled={!input.trim() || isLoading}
            >
              <Send size={14} strokeWidth={3} />
            </Button>
          </div>
          <div className="flex items-center justify-center gap-2 mt-4 text-default-400 opacity-70">
            <Info size={12} />
            <p className="text-xs leading-none">
              Manual verification required for medical findings
            </p>
          </div>
        </div>
      </footer>
    </div>
  );
}
