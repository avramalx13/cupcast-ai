import { Bot } from "lucide-react";

type Props = {
  explanation: string;
};

export function AnalystPanel({ explanation }: Props) {
  return (
    <section className="overflow-hidden rounded-lg border border-white bg-white/95 shadow-sm">
      <div className="flex items-center gap-3 border-b border-line bg-[#102f1d] px-5 py-4 text-white">
        <div className="flex h-9 w-9 items-center justify-center rounded-full bg-white text-[#102f1d]">
          <Bot aria-hidden className="h-5 w-5" />
        </div>
        <h2 className="text-lg font-black">Analyst</h2>
      </div>
      <p className="p-5 leading-7 text-slate-700">{explanation}</p>
    </section>
  );
}
