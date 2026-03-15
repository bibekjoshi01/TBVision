import { API_BASE } from "@/lib/constants";
import type { FollowUpHistoryResponse, FollowUpResponse } from "@/types";

export async function sendFollowUp(
  reportId: string,
  question: string
): Promise<FollowUpResponse> {
  const response = await fetch(`${API_BASE}/api/follow-up`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ report_id: reportId, question }),
  });

  if (!response.ok) {
    throw new Error("Failed to fetch response");
  }

  return response.json() as Promise<FollowUpResponse>;
}

export async function fetchFollowUpHistory(
  reportId: string
): Promise<FollowUpHistoryResponse> {
  const response = await fetch(
    `${API_BASE}/api/follow-up/history?report_id=${encodeURIComponent(reportId)}`
  );

  if (!response.ok) {
    throw new Error("Failed to fetch follow-up history");
  }

  return response.json() as Promise<FollowUpHistoryResponse>;
}
