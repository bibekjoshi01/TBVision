import { useState, useEffect, useCallback } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formSchema, type FormData } from "@/lib/schema";
import type { PredictionResponse } from "@/types";
import { runPrediction } from "@/services";
import { useImagePersistence } from "./useImagePersistence";

type ViewState = "form" | "results";

const STORAGE_KEY = "tbvision_latest_results";

interface PersistedState {
  data: PredictionResponse;
  viewState: ViewState;
}

export interface UseAnalysisReturn {
  viewState: ViewState;
  setViewState: React.Dispatch<React.SetStateAction<ViewState>>;
  selectedFile: File | null;
  previewUrl: string | null;
  isAnalyzing: boolean;
  error: string | null;
  results: PredictionResponse | null;
  control: ReturnType<typeof useForm<FormData>>["control"];
  errors: ReturnType<typeof useForm<FormData>>["formState"]["errors"];
  handleSubmit: ReturnType<typeof useForm<FormData>>["handleSubmit"];
  handleImageUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  removeImage: () => void;
  onFormSubmit: (data: FormData) => Promise<void>;
  resetAnalysis: () => void;
  handlePrint: () => void;
}

export function useAnalysis(): UseAnalysisReturn {
  const [viewState, setViewState] = useState<ViewState>("form");
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionResponse | null>(null);

  const {
    selectedFile,
    previewUrl,
    handleImageUpload,
    removeImage,
    restoreImage,
  } = useImagePersistence();

  const {
    control,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(formSchema) as Parameters<typeof useForm<FormData>>[0]["resolver"],
    defaultValues: {
      patient_info: { age: "", sex: "male", region: "" },
      symptoms: {
        cough: false,
        cough_duration_days: "",
        fever: false,
        night_sweats: false,
        weight_loss: false,
        chest_pain: false,
        shortness_of_breath: false,
        fatigue: false,
      },
      risk_factors: {
        smoker: false,
        diabetes: false,
        hiv_positive: false,
        close_contact_tb_patient: false,
        immunocompromised: false,
      },
      medical_history: {
        previous_tb: false,
        chronic_lung_disease: false,
        recent_pneumonia: false,
      },
      screening_context: { screening_type: "symptomatic", location: "" },
      clinical_history: "",
    },
  });

  // Restore state on mount
  useEffect(() => {
    const init = async () => {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) return;
      try {
        const parsed = JSON.parse(raw) as PersistedState;
        setResults(parsed.data);
        await restoreImage();
        if (parsed.viewState === "results") setViewState("results");
      } catch (e) {
        console.error("Failed to restore saved results", e);
      }
    };
    init();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const onFormSubmit = useCallback(
    async (data: FormData) => {
      if (!selectedFile) {
        setError("Medical image (Radiograph) is required.");
        return;
      }
      setIsAnalyzing(true);
      setError(null);
      try {
        const resData = await runPrediction(
          selectedFile,
          data as unknown as Record<string, unknown>
        );
        setResults(resData);
        setViewState("results");
        localStorage.setItem(
          STORAGE_KEY,
          JSON.stringify({ data: resData, viewState: "results" } satisfies PersistedState)
        );
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "An unexpected error occurred."
        );
      } finally {
        setIsAnalyzing(false);
      }
    },
    [selectedFile]
  );

  const resetAnalysis = useCallback(() => {
    setViewState("form");
    setResults(null);
    reset();
    localStorage.removeItem(STORAGE_KEY);
    removeImage();
  }, [reset, removeImage]);

  const handlePrint = useCallback(() => {
    window.print();
  }, []);

  return {
    viewState,
    setViewState,
    selectedFile,
    previewUrl,
    isAnalyzing,
    error,
    results,
    control,
    errors,
    handleSubmit,
    handleImageUpload,
    removeImage,
    onFormSubmit,
    resetAnalysis,
    handlePrint,
  };
}
