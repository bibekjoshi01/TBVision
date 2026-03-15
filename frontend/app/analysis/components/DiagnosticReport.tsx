import { Activity, Info, MessageCircle } from "lucide-react";
import { Button } from "@heroui/button";
import { Card, CardBody } from "@heroui/card";
import { Divider } from "@heroui/divider";
import { Image } from "@heroui/image";
import { Progress as UIProgress } from "@heroui/progress";
import { Chip as UIChip } from "@heroui/chip";
import NextLink from "next/link";
import type { PredictionResponse } from "@/types";
import { ProbabilityPie } from "./ProbabilityPie";
import MarkdownRenderer from "@/components/markdown/markdown-renderer";

const Progress = UIProgress as any;
const Chip = UIChip as any;

interface DiagnosticReportProps {
  results: PredictionResponse;
  previewUrl: string | null;
}

export function DiagnosticReport({ results, previewUrl }: DiagnosticReportProps) {
  const reportContent = results.explanation?.trim();
  const evidence = results.evidence || [];

  return (
    <div id="report-content" className="space-y-8 animate-in fade-in duration-500">
      {/* 1. Report Header Card */}
      <Card className="shadow-none border border-default-200">
        <CardBody className="p-8 space-y-8">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-6 text-left">
            <div className="flex items-center gap-4">
              <div className={`p-3 rounded-xl ${results.prediction === "TB" ? "bg-danger/10 text-danger" : "bg-success/10 text-success"}`}>
                <Activity size={32} />
              </div>
              <div>
                <h2 className="text-2xl font-black">{results.prediction_label} Detected</h2>
                <p className="text-sm text-default-500 font-medium">Diagnostic Reference: {new Date(results.metadata?.timestamp || Date.now()).toLocaleDateString()}</p>
              </div>
            </div>
            <ProbabilityPie
              probabilities={results.probabilities || {}}
              prediction={results.prediction}
            />
          </div>
        </CardBody>
      </Card>

      {/* 2. Visual AI Analysis Card */}
      <Card className="shadow-none border border-default-200">
        <CardBody className="p-8">
          <div className="space-y-4 text-left">
            <div className="space-y-2 mb-2">
              <p className="text-md font-black text-default-800 uppercase tracking-wider">Visual AI Analysis</p>
              <p className="text-sm text-default-600 font-semibold leading-relaxed">Neural activation map identifying potential clinical abnormalities.</p>
            </div>
            <div className="grid grid-cols-2 gap-4">
              <div className="space-y-2">
                <div className="aspect-square rounded-xl bg-default-50 border border-default-200 overflow-hidden">
                  <Image src={previewUrl || undefined} className="object-contain w-full h-full" />
                </div>
                <p className="text-xs text-center text-default-400 font-bold uppercase">Source Image</p>
              </div>
              <div className="space-y-2">
                <div className="aspect-square rounded-xl bg-default-50 border border-primary/20 overflow-hidden">
                  <Image src={results.gradcam_image || undefined} className="object-contain w-full h-full" />
                </div>
                <p className="text-xs text-center text-primary font-bold uppercase">Activation Map</p>
              </div>
            </div>
            <div className="bg-default-50 p-5 rounded-xl text-sm leading-relaxed text-default-700 border border-default-100 shadow-sm">
              <Info size={16} className="inline mr-2 text-primary" />
              {results.gradcam_region}
            </div>
          </div>
        </CardBody>
      </Card>

        {/* 4. Patient & Clinical Context Card */}
        <Card className="shadow-none border border-default-200">
          <CardBody className="p-8">
            <div className="space-y-4 text-left">
              <div className="space-y-2 mb-2">
                <p className="text-md font-black text-default-800 uppercase tracking-wider">Patient & Clinical Context</p>
                <p className="text-sm text-default-600 font-semibold leading-relaxed">Physiological indicators and provenance data collected during intake.</p>
              </div>
              <div className="bg-default-50 p-6 rounded-xl border border-default-100 space-y-6">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <p className="text-xs text-default-500 font-bold uppercase">Patient Info</p>
                    <p className="text-sm font-bold">{results.metadata?.patient_info?.age}y, {results.metadata?.patient_info?.sex}</p>
                  </div>
                  <div>
                    <p className="text-xs text-default-500 font-bold uppercase">Provenance</p>
                    <p className="text-sm font-bold">{results.metadata?.patient_info?.region || "N/A"}</p>
                  </div>
                </div>
                <Divider />
                <div className="space-y-4">
                  <p className="text-xs text-default-600 font-bold uppercase">Clinical Indicators</p>
                  <div className="flex flex-wrap gap-2">
                    {Object.entries(results.metadata?.symptoms || {}).map(([key, value]) => (
                      value === true && <Chip key={key} size="md" variant="flat" color="primary" className="text-xs font-bold">{key.replace("_", " ")}</Chip>
                    ))}
                    {Object.entries(results.metadata?.risk_factors || {}).map(([key, value]) => (
                      value === true && <Chip key={key} size="md" variant="flat" color="warning" className="text-xs font-bold">{key.replace("_", " ")}</Chip>
                    ))}
                  </div>
                </div>
                <Divider />
                <div className="space-y-2">
                  <p className="text-xs text-default-600 font-bold uppercase">Clinical Observations</p>
                  <p className="text-sm leading-relaxed text-default-700 italic">
                    {results.metadata?.clinical_history || "No additional observations provided."}
                  </p>
                </div>
              </div>
            </div>
          </CardBody>
        </Card>

      {/* 5. Medical Summary & Insights Card */}
      {reportContent && (
        <Card className="shadow-none border border-default-200">
          <CardBody className="p-8">
            <div className="space-y-4 text-left">
              <div className="space-y-2 mb-2">
                <p className="text-sm font-black text-default-800 uppercase tracking-wider">Medical Summary & Insights</p>
                <p className="text-xs text-default-600 font-semibold leading-relaxed">Automated synthesis of model results and clinical domain knowledge.</p>
              </div>
                <MarkdownRenderer content={reportContent} />
            </div>
          </CardBody>
        </Card>
      )}

      {/* 6. Evidence Card */}
      {evidence.length > 0 && (
        <Card className="shadow-none border border-default-200">
          <CardBody className="p-8">
            <div className="space-y-4 text-left">
              <div className="space-y-2 mb-2">
                <p className="text-sm font-black text-default-800 uppercase tracking-wider">Evidence</p>
                <p className="text-xs text-default-600 font-semibold leading-relaxed">Key model signals supporting this assessment.</p>
              </div>
              <div className="space-y-3">
                {evidence.map((item, index) => (
                  <div
                    key={`${item.label}-${index}`}
                    className="p-4 rounded-xl border border-default-200 bg-default-50"
                  >
                    <div className="flex items-center justify-between gap-4">
                      <p className="text-sm font-bold text-default-800">{item.label}</p>
                      <Chip size="md" variant="flat" color="primary" className="text-xs font-bold">
                        {(item.score * 100).toFixed(1)}%
                      </Chip>
                    </div>
                    {item.description && (
                      <p className="mt-2 text-sm text-default-600 leading-relaxed">
                        {item.description}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </CardBody>
        </Card>
      )}

      {/* 7. AI Chat CTA Card */}
      <Card className="shadow-none border border-primary/20 bg-primary/5 no-print">
        <CardBody className="p-8 flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex items-center gap-4">
            <div className="p-3 rounded-2xl bg-primary text-white shadow-lg shadow-primary/20">
              <MessageCircle size={24} />
            </div>
            <div className="text-left">
              <h3 className="text-lg font-black">Discuss Findings with AI</h3>
              <p className="text-sm text-default-600 font-medium">Have questions about these results? Chat with our specialist AI assistant.</p>
            </div>
          </div>
          <Button
            as={NextLink}
            href="/chat"
            color="primary"
            size="lg"
            className="font-bold px-8 shadow-lg shadow-primary/20"
            startContent={<MessageCircle size={18} />}
          >
            Start Conversation
          </Button>
        </CardBody>
      </Card>
    </div>
  );
}
