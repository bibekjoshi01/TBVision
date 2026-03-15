import * as z from "zod";

export const formSchema = z.object({
  patient_info: z.object({
    age: z.string().min(1, "Age is required"),
    sex: z.string(),
    region: z.string().optional(),
  }),
  symptoms: z.object({
    cough: z.boolean().default(false),
    cough_duration_days: z.string().optional(),
    fever: z.boolean().default(false),
    night_sweats: z.boolean().default(false),
    weight_loss: z.boolean().default(false),
    chest_pain: z.boolean().default(false),
    shortness_of_breath: z.boolean().default(false),
    fatigue: z.boolean().default(false),
  }),
  risk_factors: z.object({
    smoker: z.boolean().default(false),
    diabetes: z.boolean().default(false),
    hiv_positive: z.boolean().default(false),
    close_contact_tb_patient: z.boolean().default(false),
    immunocompromised: z.boolean().default(false),
  }),
  medical_history: z.object({
    previous_tb: z.boolean().default(false),
    chronic_lung_disease: z.boolean().default(false),
    recent_pneumonia: z.boolean().default(false),
  }),
  screening_context: z.object({
    screening_type: z.string().default("symptomatic"),
    location: z.string().optional(),
  }),
  clinical_history: z.string().optional(),
});

export type FormData = z.infer<typeof formSchema>;

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
  metadata: any;
  evidence?: Array<{
    label: string;
    score: number;
    description: string;
  }>;
}
