import { Link } from "@heroui/link";
import { button as buttonStyles } from "@heroui/theme";
import { title, subtitle } from "@/components/primitives";
import { Activity } from "lucide-react";

export default function Home() {
  return (
    <section className="flex flex-col items-center justify-center gap-8 py-12 md:py-24">
      <div className="inline-block max-w-3xl text-center justify-center">
        <div className="flex justify-center mb-6">
          <div className="p-4 bg-primary/10 rounded-full text-primary">
            <Activity size={48} />
          </div>
        </div>
        <h1 className={title({ size: "lg" })}>
          Advanced AI Analysis for&nbsp;
        </h1>
        <h1 className={title({ size: "lg", color: "blue" })}>
          Chest X-Rays
        </h1>
        <div className={subtitle({ class: "mt-6 text-default-500" })}>
          Empowering medical professionals with AI-driven, instantaneous insights into chest X-ray imaging. Upload, process, and diagnose with confidence.
        </div>
      </div>

      <div className="flex gap-4 mt-4">
        <Link
          className={buttonStyles({
            color: "primary",
            radius: "md",
            size: "lg",
          })}
          href="/analysis"
        >
          Upload X-ray
        </Link>
        <Link
          className={buttonStyles({
            variant: "bordered",
            radius: "md",
            size: "lg",
          })}
          href="/analysis"
        >
          View Demo
        </Link>
      </div>

      <div className="mt-16 w-full max-w-5xl rounded-2xl bg-default-50 border border-default-200 shadow-sm p-4 flex flex-col items-center justify-center">
        <div className="w-full aspect-[2/1] bg-default-100 rounded-xl flex items-center justify-center text-default-400 border border-dashed border-default-300">
           <div className="flex flex-col items-center gap-2">
             <Activity size={32} className="text-default-300" />
             <p className="text-sm font-medium">AI Analysis Visualization Area</p>
           </div>
        </div>
      </div>
    </section>
  );
}
