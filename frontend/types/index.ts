import { SVGProps } from "react";

export type IconSvgProps = SVGProps<SVGSVGElement> & {
  size?: number;
};

// ---------------------------------------------------------------------------
// Analysis / Prediction types
// ---------------------------------------------------------------------------

export interface EvidenceItem {
  label: string;
  score: number;
  description: string;
}

export interface PatientInfo {
  age: string | number;
  sex: string;
  region?: string;
}

export interface Symptoms {
  cough?: boolean;
  cough_duration_days?: string | number;
  fever?: boolean;
  night_sweats?: boolean;
  weight_loss?: boolean;
  chest_pain?: boolean;
  shortness_of_breath?: boolean;
  fatigue?: boolean;
  [key: string]: boolean | string | number | undefined;
}

export interface RiskFactors {
  smoker?: boolean;
  diabetes?: boolean;
  hiv_positive?: boolean;
  close_contact_tb_patient?: boolean;
  immunocompromised?: boolean;
  [key: string]: boolean | undefined;
}

export interface PredictionMetadata {
  timestamp?: string;
  patient_info?: PatientInfo;
  symptoms?: Symptoms;
  risk_factors?: RiskFactors;
  clinical_history?: string;
  [key: string]: unknown;
}

export interface PredictionResponse {
  prediction: string;
  prediction_label: string;
  probability: number;
  probabilities: Record<string, number>;
  uncertainty_level: string;
  gradcam_image?: string;
  gradcam_region?: string;
  summary?: string;
  explanation?: string;
  metadata: PredictionMetadata;
  evidence?: EvidenceItem[];
}

// ---------------------------------------------------------------------------
// Chat types
// ---------------------------------------------------------------------------

export type MessageRole = "user" | "assistant";

export interface Message {
  id: string;
  role: MessageRole;
  content: string;
  timestamp: Date;
}

// ---------------------------------------------------------------------------
// RAG API response
// ---------------------------------------------------------------------------

export interface RagDocument {
  text: string;
  [key: string]: unknown;
}

export interface RagResponse {
  documents?: RagDocument[];
  [key: string]: unknown;
}
