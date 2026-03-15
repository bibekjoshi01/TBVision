"use client";

import { useRef } from "react";
import { motion, useScroll, useTransform } from "framer-motion";
import { Link } from "@heroui/link";
import { Image } from "@heroui/image";
import { button as buttonStyles } from "@heroui/theme";
import { title, subtitle } from "@/components/primitives";
import { ArrowRight, ShieldCheck } from "lucide-react";

export default function Home() {
  const containerRef = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: containerRef,
    offset: ["start end", "end 80%"],
  });

  const width = useTransform(scrollYProgress, [0, 1], ["80%", "100%"]);

  return (
    <section className="relative flex flex-col items-center justify-center gap-6 pt-12 overflow-hidden">
      {/* Background ambient light */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[800px] h-[400px] bg-primary/20 blur-[120px] rounded-full pointer-events-none -z-10" />
      {/* Top Badge */}
      <div className="flex gap-2 items-center mt-6">
        <ShieldCheck size={16} className="text-success" />
        <span className="text-sm font-medium text-default-600">AI-Powered Tuberculosis Detection</span>
      </div>

      <div className="inline-block max-w-4xl text-center justify-center z-10">
        <h1 className={title({ size: "lg", class: "tracking-tight leading-tight" })}>
          <strong>Next-Generation Analysis for</strong>{" "}
        </h1>
        <h1 className={title({ size: "lg", color: "blue", class: "tracking-tight leading-tight block mt-2" })}>
          <strong>Tuberculosis Screening</strong>
        </h1>
        <div className="mt-8 text-default-500 max-w-3xl mx-auto leading-relaxed text-lg font-body">
          Detect tuberculosis from chest X-rays using AI-powered analysis.
          Get instant predictions, visual explanations, and clinical insights to support early diagnosis.
        </div>
      </div>

      <div className="flex flex-col sm:flex-row gap-4 mt-8 z-10 w-full sm:w-auto px-4">
        <Link
          className={buttonStyles({
            color: "primary",
            radius: "full",
            size: "lg",
            class: "font-semibold w-full sm:w-auto px-8"
          })}
          href="/analysis"
        >
          Start Analysis
          <ArrowRight size={18} className="ml-2" />
        </Link>
        <Link
          className={buttonStyles({
            variant: "bordered",
            radius: "full",
            size: "lg",
            class: "font-semibold bg-background hover:bg-default-100 w-full sm:w-auto px-8"
          })}
          href="/demo"
        >
          See How It Works
        </Link>
      </div>

      {/* Feature showcase mockup area */}
      <div ref={containerRef} className="mt-20 w-full max-w-6xl z-10 px-4 flex justify-center pb-16">
        <motion.div
          style={{ width }}
          className="relative w-full rounded-2xl bg-content1/50 backdrop-blur-xl border border-default-200 shadow-2xl p-2 md:p-4 rings-1 ring-white/10"
        >
          {/* macOS style window dots */}
          <div className="flex gap-2 mb-4 px-2 pt-2">
            <div className="w-3 h-3 rounded-full bg-danger-400" />
            <div className="w-3 h-3 rounded-full bg-warning-400" />
            <div className="w-3 h-3 rounded-full bg-success-400" />
          </div>
          <div className="w-full aspect-video bg-default-50/50 rounded-xl flex items-center justify-center text-default-400 border border-default-200 overflow-hidden relative">
            <Image
              src="/analysis.png"
              fallbackSrc="https://nextui.org/images/hero-card-complete.jpeg"
              alt="TBVision Dashboard Screenshot"
              className="w-full h-full object-cover"
              radius="none"
              classNames={{
                wrapper: "w-full h-full max-w-none"
              }}
            />
          </div>
        </motion.div>
      </div>
    </section>
  );
}
