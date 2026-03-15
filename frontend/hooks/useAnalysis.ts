import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formSchema } from "@/lib/schema";
import type { FormData } from "@/lib/schema";
import type { PredictionResponse } from "@/types";
import { runPrediction } from "@/services/predict";
import { saveFile, getFile, clearStorage } from "@/lib/db";
import { normalizeMetadata } from "@/lib/metadata";

export function useAnalysis() {
  const [viewState, setViewState] = useState<"form" | "results">("form");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionResponse | null>(null);

  const {
    control,
    handleSubmit,
    reset,
    formState: { errors },
  } = useForm<FormData>({
    resolver: zodResolver(formSchema) as any,
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
      screening_context: {
        screening_type: "symptomatic",
        location: "",
      },
      clinical_history: "",
    },
  });

  useEffect(() => {
    const init = async () => {
      const savedResults = localStorage.getItem("tbvision_latest_results");
      if (savedResults) {
        try {
          const parsed = JSON.parse(savedResults);
          const data = parsed.data as PredictionResponse;
          setResults(data);

          // Restore image from IndexedDB
          const file = await getFile("latest_upload");
          if (file) {
            setSelectedFile(file);
            setPreviewUrl(URL.createObjectURL(file));
          }

          if (parsed.viewState === "results") setViewState("results");
        } catch (e) {
          console.error("Failed to parse saved results", e);
        }
      }
    };
    init();
  }, []);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreviewUrl(URL.createObjectURL(file));
      setError(null);
      // Persist to IDB
      saveFile("latest_upload", file).catch(console.error);
    }
  };

  const removeImage = () => {
    setPreviewUrl(null);
    setSelectedFile(null);
    clearStorage().catch(console.error);
  };

  const onFormSubmit = async (data: any) => {
    if (!selectedFile) {
      setError("Medical image (X-Ray) is required.");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    try {
      const resData = await runPrediction(selectedFile, normalizeMetadata(data));
      setResults(resData as PredictionResponse);
      setViewState("results");
      localStorage.setItem(
        "tbvision_latest_results",
        JSON.stringify({ data: resData, viewState: "results" })
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "An unexpected error occurred.");
    } finally {
      setIsAnalyzing(false);
    }
  };

  const resetAnalysis = () => {
    setViewState("form");
    setResults(null);
    setSelectedFile(null);
    setPreviewUrl(null);
    reset();
    localStorage.removeItem("tbvision_latest_results");
    clearStorage().catch(console.error);
  };

  const handlePrint = () => {
    window.print();
  };

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
