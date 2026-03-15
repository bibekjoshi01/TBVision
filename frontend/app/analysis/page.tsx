"use client";

import { useState } from "react";
import { Button } from "@heroui/button";
import { Input, Textarea } from "@heroui/input";
import { Card, CardBody, CardHeader } from "@heroui/card";
import { Divider } from "@heroui/divider";
import { Image } from "@heroui/image";
import { title } from "@/components/primitives";
import { UploadCloud, FileText, Download } from "lucide-react";

export default function AnalysisPage() {
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [results, setResults] = useState<{
    originalUrl: string;
    gradCamUrl: string;
    summary: string;
    probabilities: Record<string, number>;
  } | null>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setSelectedImage(url);
      setResults(null); 
    }
  };

  const handleAnalyze = () => {
    if (!selectedImage) return;
    setIsAnalyzing(true);
    // Mock analysis delay
    setTimeout(() => {
      setResults({
        originalUrl: selectedImage,
        gradCamUrl: selectedImage, // Mock grad-cam uses the same image in real life it will be a different image
        summary: "High probability of Tuberculosis detected. Please review Grad-CAM heatmap for focal areas.",
        probabilities: {
          "Tuberculosis": 0.92,
          "Normal": 0.08,
        }
      });
      setIsAnalyzing(false);
    }, 2000);
  };

  const handleDownloadReport = async () => {
    try {
      const element = document.getElementById("report-content");
      if (!element) return;
      
      const html2canvas = (await import("html2canvas")).default;
      const { jsPDF } = await import("jspdf");

      const canvas = await html2canvas(element, { scale: 2 });
      const imgData = canvas.toDataURL("image/png");
      const pdf = new jsPDF({
        orientation: "portrait",
        unit: "mm",
        format: "a4"
      });
      
      const pdfWidth = pdf.internal.pageSize.getWidth();
      const pdfHeight = (canvas.height * pdfWidth) / canvas.width;
      
      pdf.addImage(imgData, 'PNG', 0, 0, pdfWidth, pdfHeight);
      pdf.save("TBVision-Analysis-Report.pdf");
    } catch (error) {
      console.error("Error generating PDF", error);
      alert("Failed to generate PDF report.");
    }
  };

  return (
    <div className="flex flex-col gap-6 pb-8">
      <div>
        <h1 className={title({ size: "sm" })}>Analysis Workspace</h1>
        <p className="text-default-500 mt-2">Upload a chest X-ray and optionally provide patient metadata to begin AI analysis.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column: Inputs */}
        <div className="flex flex-col gap-6 lg:col-span-1">
          <Card className="p-2 shadow-sm border border-default-200">
            <CardHeader className="flex gap-3">
              <UploadCloud className="text-primary" />
              <div className="flex flex-col">
                <p className="text-md font-semibold">Image Upload</p>
                <p className="text-small text-default-500">Supported: JPG, PNG</p>
              </div>
            </CardHeader>
            <Divider />
            <CardBody className="flex flex-col gap-4">
              <div className="border-2 border-dashed border-default-300 rounded-lg p-8 flex flex-col items-center justify-center text-center gap-2 hover:border-primary transition-colors bg-default-50">
                <input 
                  type="file" 
                  accept="image/*" 
                  className="hidden" 
                  id="file-upload" 
                  onChange={handleImageUpload} 
                />
                <label htmlFor="file-upload" className="cursor-pointer flex flex-col items-center gap-2 w-full">
                  <UploadCloud size={32} className="text-default-400" />
                  <span className="text-default-600 font-medium">Click to browse or drag and drop</span>
                </label>
              </div>
              {selectedImage && (
                <div className="relative aspect-square w-full rounded-md overflow-hidden bg-black/5 border border-default-200">
                  <Image 
                    src={selectedImage} 
                    alt="Selected X-Ray" 
                    className="object-cover w-full h-full"
                  />
                </div>
              )}
            </CardBody>
          </Card>

          <Card className="p-2 shadow-sm border border-default-200">
            <CardHeader className="flex gap-3">
              <FileText className="text-primary" />
              <div className="flex flex-col">
                <p className="text-md font-semibold">Patient Metadata</p>
                <p className="text-small text-default-500">Optional clinical notes</p>
              </div>
            </CardHeader>
            <Divider />
            <CardBody className="flex flex-col gap-4">
              <Input label="Patient ID" placeholder="e.g. PID-12345" variant="bordered" />
              <Input label="Age" type="number" placeholder="e.g. 45" variant="bordered" />
              <Textarea 
                label="Clinical History" 
                placeholder="Brief clinical notes..." 
                variant="bordered"
                minRows={5}
              />
              
              <Button 
                color="primary" 
                className="mt-4 w-full" 
                size="lg"
                isDisabled={!selectedImage}
                isLoading={isAnalyzing}
                onPress={handleAnalyze}
              >
                {isAnalyzing ? "Processing AI..." : "Analyze X-Ray"}
              </Button>
            </CardBody>
          </Card>
        </div>

        {/* Right Column: Outputs */}
        <div className="flex flex-col gap-6 lg:col-span-2">
           {results ? (
             <Card className="p-2 shadow-sm border border-default-200 h-full">
               <CardHeader className="flex justify-between items-center">
                 <div className="flex flex-col">
                   <p className="text-lg font-semibold text-primary">Analysis Results</p>
                   <p className="text-small text-default-500">Generated successfully</p>
                 </div>
                 <Button 
                   color="secondary" 
                   variant="flat" 
                   startContent={<Download size={18} />}
                   onPress={handleDownloadReport}
                 >
                   Export PDF
                 </Button>
               </CardHeader>
               <Divider />
               <CardBody className="flex flex-col gap-6" id="report-content">
                 
                 <div className="bg-danger/10 p-4 rounded-lg border border-danger/20 flex flex-col gap-2">
                   <h3 className="font-semibold text-danger-600">AI Conclusion</h3>
                   <p className="text-danger-800">{results.summary}</p>
                 </div>

                 <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                   <div className="flex flex-col gap-2">
                     <p className="font-medium text-default-700">Original X-Ray</p>
                     <div className="relative aspect-[4/3] rounded-lg overflow-hidden bg-black/5 border border-default-200 flex justify-center items-center">
                       <Image src={results.originalUrl} alt="Original" className="max-h-full object-contain" />
                     </div>
                   </div>
                   <div className="flex flex-col gap-2">
                     <p className="font-medium text-default-700">Grad-CAM Heatmap</p>
                     <div className="relative aspect-[4/3] rounded-lg overflow-hidden bg-black/5 border border-default-200 flex justify-center items-center">
                       <div className="relative w-full h-full flex justify-center items-center">
                         <Image 
                           src={results.gradCamUrl} 
                           alt="Grad-CAM" 
                           className="max-h-full object-contain grayscale" 
                         />
                         <div className="absolute inset-0 bg-gradient-to-tr from-transparent to-red-500/30 mix-blend-overlay z-10 pointer-events-none" />
                       </div>
                     </div>
                   </div>
                 </div>
               </CardBody>
             </Card>
           ) : (
             <div className="h-full min-h-[400px] border-2 border-dashed border-default-200 rounded-xl flex flex-col items-center justify-center bg-default-50 text-default-400 p-6 text-center">
               <UploadCloud size={48} className="mb-4 text-default-300" />
               <p className="text-lg">Waiting for Input</p>
               <p className="text-sm mt-2 max-w-sm">Upload an X-ray image and click "Analyze X-Ray" to view the AI results here.</p>
             </div>
           )}
        </div>
      </div>
    </div>
  );
}
