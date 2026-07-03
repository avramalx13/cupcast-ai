"use client";

import Link from "next/link";
import type { ComponentType, ReactNode } from "react";
import { BarChart3, Bot, Home, ShieldCheck, Trophy, Zap } from "lucide-react";
import { WorldCupMark } from "@/components/WorldCupMark";

const navItems = [
  { href: "/", label: "Home", icon: Home },
  { href: "/predictions", label: "Predict", icon: Zap },
  { href: "/simulator", label: "Simulate", icon: Trophy },
  { href: "/full-tournament", label: "Full Cup", icon: Trophy },
  { href: "/analyst", label: "Analyst", icon: Bot },
  { href: "/backtesting", label: "Backtest", icon: BarChart3 }
];

type ShellProps = {
  children: ReactNode;
};

type HeroProps = {
  title: string;
  eyebrow: string;
  description: string;
  icon: ComponentType<{ className?: string; "aria-hidden"?: boolean }>;
  aside?: ReactNode;
  children?: ReactNode;
};

export function AppShell({ children }: ShellProps) {
  return (
    <main className="app-frame">
      <header className="app-topbar px-4 py-3 text-white">
        <div className="mx-auto flex max-w-7xl items-center gap-4 pr-0 sm:pr-56">
          <Link href="/" className="flex shrink-0 items-center gap-3">
            <WorldCupMark compact />
            <span className="hidden text-sm font-black uppercase text-white/75 lg:inline">CupCast AI</span>
          </Link>
          <nav className="flex min-w-0 flex-1 items-center justify-end gap-1 overflow-x-auto">
            {navItems.map(({ href, label, icon: Icon }) => (
              <Link
                key={href}
                href={href}
                className="app-nav-link"
              >
                <Icon aria-hidden className="h-4 w-4" />
                <span>{label}</span>
              </Link>
            ))}
          </nav>
        </div>
      </header>
      {children}
    </main>
  );
}

export function PageHero({ title, eyebrow, description, icon: Icon, aside, children }: HeroProps) {
  return (
    <section className="page-hero px-6">
      <div className="hero-grid" />
      <div className="hero-sweep" />
      <div className="relative mx-auto grid max-w-7xl gap-8 py-16 text-center md:grid-cols-[minmax(0,1fr)_22rem] md:items-end">
        <div className="mx-auto max-w-3xl">
          <div className="hero-badge">
            <Icon aria-hidden className="h-4 w-4" />
            {eyebrow}
          </div>
          <h1 className="mt-5 text-4xl font-black leading-tight text-white text-balance md:text-6xl">{title}</h1>
          <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-white/70">{description}</p>
          {children ? <div className="mt-7 flex flex-wrap justify-center gap-3">{children}</div> : null}
        </div>
        {aside ? (
          <div className="hero-panel p-5">
            {aside}
          </div>
        ) : (
          <div className="hero-panel hidden p-5 md:block">
            <ShieldCheck aria-hidden className="h-5 w-5 text-emerald-200" />
            <p className="mt-3 text-sm leading-6 text-white/70">
              Real data, explicit baselines, and visible model limitations.
            </p>
          </div>
        )}
      </div>
    </section>
  );
}
