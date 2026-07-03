type Props = {
  compact?: boolean;
  className?: string;
};

const markSrc = "/brand/cupcast-header-mark-art.png";

export function WorldCupMark({ compact = false, className = "" }: Props) {
  if (compact) {
    return (
      <div
        className={`inline-flex items-center rounded-full border border-white/25 bg-white/10 p-2 text-white shadow-lg shadow-black/20 backdrop-blur ${className}`}
      >
        <div className="flex h-11 w-11 items-center justify-center overflow-hidden rounded-full bg-[#07120d] ring-1 ring-cyan-300/40">
          <img src={markSrc} alt="CupCast AI mark" className="h-full w-full scale-125 object-contain" />
        </div>
      </div>
    );
  }

  return (
    <div
      className={`inline-flex items-center gap-3 rounded-full border border-white/25 bg-white/10 px-3 py-2 text-white shadow-lg shadow-black/20 backdrop-blur ${className}`}
    >
      <div className="flex h-11 w-11 items-center justify-center overflow-hidden rounded-full bg-[#07120d] ring-1 ring-cyan-300/40">
        <img src={markSrc} alt="CupCast AI mark" className="h-full w-full scale-125 object-contain" />
      </div>
      <div className="leading-none">
        <div className="text-2xl font-black">CupCast AI</div>
        <div className="mt-1 text-[0.62rem] font-bold uppercase text-white/70">
          Forecast engine
        </div>
      </div>
    </div>
  );
}
