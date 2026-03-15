"use client";

import { Button } from "@heroui/button";
import { RotateCcw, Printer } from "lucide-react";
import { title } from "@/components/primitives";
import { useAnalysis } from "@/hooks/useAnalysis";
import { AnalysisForm } from "./components/AnalysisForm";
import { DiagnosticReport } from "./components/DiagnosticReport";

export default function AnalysisPage() {
  const {
    viewState,
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
  } = useAnalysis();

  return (
    <div className="max-w-5xl mx-auto px-4 pb-8">
      {/* Page Title */}
      <div className="flex justify-between items-end mb-8 border-b border-default-100 pb-6">
        <div className="flex flex-col gap-1">
          <span className="text-[10px] font-black text-primary uppercase tracking-[0.1em] mb-1">Next-Generation Analysis</span>
          <h1 className={title({ size: "md", class: "leading-tight" })}>
            <strong>{viewState === "form" ? "Chest X-Ray Analysis" : "Diagnostic Report"}</strong>
          </h1>
          <p className="text-default-500 text-sm mt-1 font-medium">
            {viewState === "form"
              ? "Clinical metadata collection and AI-powered screening"
              : "Consolidated results from TBVision ensemble model"}
          </p>
        </div>
        {viewState === "results" && (
          <div className="flex gap-2 no-print">
            <Button variant="flat" color="primary" size="sm" startContent={<RotateCcw size={16} />} onPress={resetAnalysis}>
              New Entry
            </Button>
            <Button variant="flat" color="primary" size="sm" startContent={<Printer size={16} />} onPress={handlePrint}>
              Print Report
            </Button>
          </div>
        )}
      </div>

      {viewState === "form" ? (
        <AnalysisForm
          control={control}
          errors={errors}
          previewUrl={previewUrl}
          isAnalyzing={isAnalyzing}
          error={error}
          handleImageUpload={handleImageUpload}
          removeImage={removeImage}
          onSubmit={handleSubmit(onFormSubmit)}
        />
      ) : (
        results && (
          <DiagnosticReport
            results={results}
            previewUrl={previewUrl}
          />
        )
      )}
    </div>
  );
}
