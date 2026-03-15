import { useState, useEffect } from "react";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { formSchema } from "./schema";
import type { FormData, PredictionResponse } from "./schema";

const DB_NAME = "TBVisionStorage";
const STORE_NAME = "images";

async function openDB() {
  return new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, 1);
    request.onupgradeneeded = () => {
      request.result.createObjectStore(STORE_NAME);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function saveFile(key: string, file: File) {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  tx.objectStore(STORE_NAME).put(file, key);
  return new Promise((resolve, reject) => {
    tx.oncomplete = resolve;
    tx.onerror = () => reject(tx.error);
  });
}

async function getFile(key: string): Promise<File | null> {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readonly");
  const request = tx.objectStore(STORE_NAME).get(key);
  return new Promise((resolve, reject) => {
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
}

async function clearStorage() {
  const db = await openDB();
  const tx = db.transaction(STORE_NAME, "readwrite");
  tx.objectStore(STORE_NAME).clear();
}

export function useAnalysis() {
  const [viewState, setViewState] = useState<"form" | "results">("form");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [results, setResults] = useState<PredictionResponse | null>(null);

  const { control, handleSubmit, reset, formState: { errors } } = useForm<FormData>({
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
        fatigue: false
      },
      risk_factors: {
        smoker: false,
        diabetes: false,
        hiv_positive: false,
        close_contact_tb_patient: false,
        immunocompromised: false
      },
      medical_history: {
        previous_tb: false,
        chronic_lung_disease: false,
        recent_pneumonia: false
      },
      screening_context: {
        screening_type: "symptomatic",
        location: ""
      },
      clinical_history: ""
    }
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

  const removeImage = (e?: any) => {
    setPreviewUrl(null);
    setSelectedFile(null);
    clearStorage().catch(console.error);
  };

  const onFormSubmit = async (data: any) => {
    if (!selectedFile) {
      setError("Medical image (Radiograph) is required.");
      return;
    }
    setIsAnalyzing(true);
    setError(null);
    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("metadata", JSON.stringify(data));
    try {
      const response = await fetch("http://127.0.0.1:8000/api/predict", { method: "POST", body: formData });
      if (!response.ok) throw new Error(`Analysis failed: ${response.statusText}`);
      const resData = await response.json();
      setResults(resData);
      setViewState("results");
      localStorage.setItem("tbvision_latest_results", JSON.stringify({ data: resData, viewState: "results" }));
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
    handlePrint
  };
}
