import { NOT_MEASURED, Row, Rows } from "./bits"
import { fmt } from "../lib/format"
import { t, tn } from "../lib/i18n"
import type { Metrics, ModelInfo } from "../lib/metrics"

const gbytes = (b: number) => `${(b / 1e9).toFixed(b >= 1e10 ? 1 : 2)} GB`
const types = (ts: string[]) => (ts.length <= 2 ? ts.join(", ") : t("{list} +{n} more", { list: ts.slice(0, 2).join(", "), n: ts.length - 2 }))

/** The model that is running: its full name, where it came from, and how each part of it is quantized. All of it is
 *  read from the GGUF headers the engine loaded; the bits per weight are the files' own bytes over the weights. */
export function ModelFacts({ info, kv }: { info: ModelInfo | null | undefined; kv?: string | number | null }) {
  if (!info) return <p className="text-ink-2">{t("Not measured: the model's files could not be read (a model that is not a GGUF has no header to show).")}</p>
  return (
    <>
      <Rows>
        <Row k={t("Model")} v={info.name || info.basename || NOT_MEASURED} />
        {info.source && <Row k={t("From")} v={info.source} />}
        <Row k={t("Architecture")} v={info.architecture ? <span className="num">{info.architecture}{info.size_label ? ` · ${info.size_label}` : ""}</span> : NOT_MEASURED} />
        <Row k="Quantization" hint={t("file variant")} v={info.variant ? <span className="num">{info.variant}</span> : NOT_MEASURED} />
        <Row k={t("On average")} v={info.bpw != null ? <span className="num">{t("{bpw} bits per weight · {size}", { bpw: info.bpw.toFixed(2), size: gbytes(info.bytes) })}</span> : NOT_MEASURED} />
        {kv != null && kv !== "" && <Row k="KV cache" v={<span className="num">{String(kv)}</span>} />}
      </Rows>
      <div className="mt-3 text-[12px] text-ink-2">{t("Quantization of each part")}</div>
      <div className="mt-1">
        {info.roles.map((r) => (
          <div key={r.role} className="flex flex-wrap items-baseline justify-between gap-x-4 border-b border-line py-1.5 last:border-0">
            <span className="text-ink-2"><span className="inline-block first-letter:uppercase">{r.role === "ple table" ? "PLE table" : r.role}</span><span className="ml-2 text-[12px] text-ink-3">{tn(r.tensors, "{count} tensor", "{count} tensors", { count: fmt(r.tensors) })}</span></span>
            <span className="num text-right" title={r.types.join(", ")}>
              {types(r.types)}<span className="text-ink-2"> · {r.bpw != null ? `${r.bpw.toFixed(2)} bpw` : "–"} · {gbytes(r.bytes)}</span>
            </span>
          </div>
        ))}
      </div>
    </>
  )
}

export const modelLine = (m: Metrics) => {
  const i = m.model_info
  const quant = i ? [i.variant, i.bpw != null ? t("{bpw} bits per weight", { bpw: i.bpw.toFixed(1) }) : null].filter(Boolean).join(" · ") : null
  return [i?.name || String(m.engine.model ?? ""), quant, m.engine.max_context ? `${fmt(Number(m.engine.max_context) / 1024)}K context` : null].filter(Boolean).join(" · ")
}
