export default function DemoPage() {
  return (
    <div className="flex flex-col items-center justify-center px-4">

      <div className="w-full max-w-6xl aspect-video bg-default-100 rounded-2xl border border-default-200 shadow-xl overflow-hidden flex items-center justify-center relative mt-8">

        {/* Placeholder while iframe loads */}
        <div className="absolute inset-0 flex flex-col items-center justify-center text-default-400 pointer-events-none">
          <svg
            className="w-16 h-16 mb-4 text-primary/50"
            fill="currentColor"
            viewBox="0 0 24 24"
          >
            <path d="M8 5v14l11-7z" />
          </svg>
          <p className="font-medium tracking-wide">
            Demo Video Initializing...
          </p>
        </div>

        {/* YouTube Embedded Video */}
        <iframe
          className="w-full h-full z-10"
          src="https://www.youtube.com/embed/1Haj5EFSahw?autoplay=1&mute=1&loop=1&playlist=1Haj5EFSahw"
          title="TBVision AI Demo"
          allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
          allowFullScreen
        ></iframe>

      </div>
    </div>
  );
}