import { API_BASE } from "@/lib/constants";
import type { RagResponse } from "@/types";

export async function sendChatQuery(
  question: string,
  topK: number = 3
): Promise<RagResponse> {
  const response = await fetch(`${API_BASE}/api/rag`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, top_k: topK }),
  });

  if (!response.ok) {
    throw new Error("Failed to fetch response");
  }

  return response.json() as Promise<RagResponse>;
}
