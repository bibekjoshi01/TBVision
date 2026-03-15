import { title } from "@/components/primitives";

export default function DemoPage() {
  return (
    <div className="flex flex-col items-center justify-center px-4">
      <h1 className={title()}>Demo</h1>
      <p className="text-default-500 text-lg text-center max-w-2xl">
        Watch how TBVision AI analyzes chest radiographs in real-time.
      </p>
      
      <div className="w-full max-w-4xl aspect-video bg-default-100 rounded-2xl border border-default-200 shadow-xl overflow-hidden flex items-center justify-center relative mt-8">
        {/* Placeholder content behind the video in case it doesn't load immediately */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-default-400">
          <svg className="w-16 h-16 mb-4 text-primary/50" fill="currentColor" viewBox="0 0 24 24">
             <path d="M8 5v14l11-7z" />
          </svg>
          <p className="font-medium tracking-wide">Demo Video Initializing...</p>
        </div>
        
        {/* Actual video element. Add your video source here */}
        <video 
          className="w-full h-full object-cover z-10 bg-black/10" 
          controls 
          autoPlay
          muted
          loop
          playsInline
        >
          {/* Uncomment and replace with actual video path when ready: */}
          {/* <source src="/demo-video.mp4" type="video/mp4" /> */}
          Your browser does not support the video tag.
        </video>
      </div>
    </div>
  );
}
