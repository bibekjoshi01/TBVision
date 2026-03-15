import { Activity, AlertCircle } from "lucide-react";
import { Button } from "@heroui/button";
import { Card, CardBody } from "@heroui/card";
import { Divider } from "@heroui/divider";
import { Textarea } from "@heroui/input";
import { Controller } from "react-hook-form";
import { RadiographAcquisition } from "./RadiographAcquisition";
import { PatientDemographics } from "./PatientDemographics";
import { PathologicalAssessment } from "./PathologicalAssessment";

interface AnalysisFormProps {
  control: any;
  errors: any;
  previewUrl: string | null;
  isAnalyzing: boolean;
  error: string | null;
  handleImageUpload: (e: React.ChangeEvent<HTMLInputElement>) => void;
  removeImage: (e?: any) => void;
  onSubmit: (e?: React.BaseSyntheticEvent) => Promise<void>;
}

export function AnalysisForm({
  control,
  errors,
  previewUrl,
  isAnalyzing,
  error,
  handleImageUpload,
  removeImage,
  onSubmit
}: AnalysisFormProps) {
  return (
    <form onSubmit={onSubmit} className="space-y-6">
      <Card className="shadow-none border border-default-200">
        <CardBody className="p-6 space-y-8">
          
          {/* Sequential Vertical Sections (1, 2, 3) */}
          <div className="space-y-10">
            <RadiographAcquisition 
              previewUrl={previewUrl} 
              handleImageUpload={handleImageUpload} 
              removeImage={removeImage} 
            />
            
            <Divider />
            
            <PatientDemographics control={control} errors={errors} />
          </div>

          <Divider />

          <PathologicalAssessment control={control} />

          <Divider />

          {/* History & Action */}
          <div className="space-y-6">
            <Controller
              name="clinical_history"
              control={control}
              render={({ field }) => (
                <Textarea {...field} minRows={10} label="Additional Clinical Observations" variant="bordered" size="md" placeholder="Record unusual findings..." />
              )}
            />

            {error && (
              <div className="bg-danger-50 p-4 rounded-xl flex items-center gap-2 text-danger text-xs font-bold">
                <AlertCircle size={16} /> {error}
              </div>
            )}

            <Button 
              type="submit" 
              color="primary" 
              size="lg" 
              className="w-full font-bold"
              isLoading={isAnalyzing}
            >
              {isAnalyzing ? "Analyzing..." : "Run Analysis"}
            </Button>
          </div>

        </CardBody>
      </Card>
    </form>
  );
}
