import type { ReviewRoute } from "../types/api";

export function formatConfidence(confidence: number): string {
  return `${(confidence * 100).toFixed(1)}%`;
}

const REVIEW_ROUTE_LABELS: Record<ReviewRoute, string> = {
  "": "—",
  accept: "Accept",
  llm_judge: "LLM Judge",
  human_review: "Human Review",
};

export function reviewRouteLabel(route: ReviewRoute): string {
  return REVIEW_ROUTE_LABELS[route];
}
