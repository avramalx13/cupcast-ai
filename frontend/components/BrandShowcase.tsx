"use client";

import { Bot, ShieldCheck, Zap } from "lucide-react";
import { useEffect, useState } from "react";

const mascots = [
  {
    src: "/brand/mascot-analyst-art.png",
    fallbackSrc: "/brand/mascot-analyst.svg",
    alt: "CupCast AI robot owl analyst mascot",
    name: "Analyst",
    role: "Model reasoning",
    icon: Bot
  },
  {
    src: "/brand/mascot-striker-art.png",
    fallbackSrc: "/brand/mascot-striker.svg",
    alt: "CupCast AI lightning fox striker mascot",
    name: "Striker",
    role: "Attacking signal",
    icon: Zap
  },
  {
    src: "/brand/mascot-keeper-art.png",
    fallbackSrc: "/brand/mascot-keeper.svg",
    alt: "CupCast AI shield bear keeper mascot",
    name: "Keeper",
    role: "Risk control",
    icon: ShieldCheck
  }
];

const assets = mascots.flatMap((mascot) => [mascot.src, mascot.fallbackSrc]).filter(Boolean);

export function BrandShowcase() {
  const [available, setAvailable] = useState<Record<string, boolean>>({});
  const [hidden, setHidden] = useState<Record<string, boolean>>({});

  useEffect(() => {
    let cancelled = false;
    Promise.all(
      assets.map(async (src) => {
        try {
          const response = await fetch(src, { method: "HEAD", cache: "no-store" });
          return [src, response.ok] as const;
        } catch {
          return [src, false] as const;
        }
      })
    ).then((entries) => {
      if (!cancelled) setAvailable(Object.fromEntries(entries));
    });
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <div className="mx-auto w-full max-w-5xl text-center">
      <div className="mx-auto grid max-w-4xl gap-3 sm:grid-cols-3">
        {mascots.map(({ src, fallbackSrc, alt, name, role, icon: Icon }) => {
          const displaySrc = imageSource(src, fallbackSrc, available, hidden);
          return (
            <div key={src} className="overflow-hidden rounded-lg border border-white/14 bg-white/[0.07]">
              <div className="h-1 bg-gradient-to-r from-[#38BDF8] via-[#22D3EE] to-[#A3E635]" />
              {displaySrc ? (
                <img
                  src={displaySrc}
                  alt={alt}
                  className="aspect-square w-full bg-white object-contain p-2"
                  onError={() => setHidden((current) => ({ ...current, [displaySrc]: true }))}
                />
              ) : (
                <div className="flex aspect-square flex-col items-center justify-center p-3 text-center">
                  <Icon aria-hidden className="h-8 w-8 text-cyan-200" />
                  <div className="mt-3 text-sm font-black text-white">{name}</div>
                </div>
              )}
              <div className="border-t border-white/10 px-3 py-2 text-center">
                <div className="text-sm font-black text-white">{name}</div>
                <div className="mt-1 text-[0.68rem] font-bold uppercase tracking-wide text-white/48">{role}</div>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function imageSource(
  src: string,
  fallbackSrc: string,
  available: Record<string, boolean>,
  hidden: Record<string, boolean>
) {
  if (available[src] && !hidden[src]) return src;
  if (fallbackSrc && available[fallbackSrc] && !hidden[fallbackSrc]) return fallbackSrc;
  return "";
}
